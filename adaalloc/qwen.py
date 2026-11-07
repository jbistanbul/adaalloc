"""Qwen3-VL reasoner (transformers): the planner, refiner and answer calls and their visual preprocessing."""

from __future__ import annotations

import math
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .config import Config
from .data import AnswerPrediction, FrameReference, QuestionSample, RetrievedSegment, VisualEvidenceInput
from .planning import (
    AdaptiveMemoryRefinement,
    AdaptiveOverviewPlan,
    fallback_answer_prediction,
    fallback_memory_refinement,
    fallback_overview_plan,
    parse_answer_prediction,
    parse_memory_refinement_v6,
    parse_overview_plan,
)
from .prompts import (
    build_grounded_structured_answer_prompt,
    build_memory_refinement_prompt_v6,
    build_overview_planning_prompt,
    build_simple_answer_prompt,
    render_frame_evidence,
    render_video_stream_evidence,
)

_REQUEST_METADATA_KEY = "_adaalloc_metadata"
_REFINEMENT_RAW_OUTPUT_CHARS = 4000
ANSWER_PROMPT_STYLES = ("grounded_structured_mcq", "simple_mcq")


@dataclass(frozen=True, slots=True)
class QwenVLProfile:
    family: str
    default_model_name: str
    patch_size: int
    merge_size: int = 2
    temporal_patch_size: int = 2

    @property
    def spatial_factor(self) -> int:
        return self.patch_size * self.merge_size

    @property
    def min_frame_pixels(self) -> int:
        return self.spatial_factor * self.spatial_factor


QWEN3_VL_PROFILE = QwenVLProfile(family="qwen3_vl", default_model_name="Qwen/Qwen3-VL-8B-Instruct", patch_size=16)


def resolve_qwen_profile(model_name: str) -> QwenVLProfile:
    """Return the processor geometry of a Qwen3-VL checkpoint (the only supported family)."""

    normalized = model_name.lower().replace("_", "-")
    if "qwen2.5-vl" in normalized or "qwen2-5-vl" in normalized:
        raise ValueError(f"Only Qwen3-VL checkpoints are supported, got {model_name!r}.")
    return QWEN3_VL_PROFILE


def resize_size_for_profile(
    profile: QwenVLProfile,
    *,
    frame_count: int,
    height: int,
    width: int,
    max_total_pixels: int,
    min_total_pixels: int | None = None,
    temporal_factor: int | None = None,
) -> tuple[int, int]:
    """Return the Qwen3-VL video-processor resize target for a total-pixel budget across ``frame_count`` frames."""

    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

    frame_count = max(1, int(frame_count))
    max_total_pixels = max(profile.min_frame_pixels, int(max_total_pixels))
    if min_total_pixels is None:
        min_total_pixels = profile.min_frame_pixels * frame_count
    min_total_pixels = max(profile.min_frame_pixels, int(min_total_pixels))
    return smart_resize(
        num_frames=frame_count,
        height=height,
        width=width,
        temporal_factor=temporal_factor or profile.temporal_patch_size,
        factor=profile.spatial_factor,
        min_pixels=min_total_pixels,
        max_pixels=max_total_pixels,
    )


def visual_tokens_from_grid_row(row: list[int] | tuple[int, int, int], profile: QwenVLProfile) -> int:
    t, h, w = [int(value) for value in row]
    return (t * h * w) // (profile.merge_size * profile.merge_size)


def _fallback_initial_queries(sample: QuestionSample) -> list[str]:
    question = sample.question_text.strip()
    return [question or "the event"]


def _read_qwen_video_with_fallback_decoder(ele: dict[str, Any], vision_process) -> tuple[Any, dict[str, Any], float]:
    import torch

    from .video import decode_frames_at_indices, probe_video_metadata

    video_path = str(ele["video"])
    if video_path.startswith("file://"):
        video_path = video_path[7:]

    metadata = probe_video_metadata(video_path)
    if metadata.total_frames is None or metadata.fps is None:
        raise FileNotFoundError(f"Unable to determine video metadata for {video_path}")

    source_total_frames = int(metadata.total_frames)
    source_fps = float(metadata.fps)
    start_frame, end_frame, ranged_total_frames = vision_process.calculate_video_frame_range(
        ele,
        source_total_frames,
        source_fps,
    )
    nframes = vision_process.smart_nframes(ele, total_frames=ranged_total_frames, video_fps=source_fps)
    idx = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    frames = decode_frames_at_indices(video_path, [int(index) for index in idx])
    if not frames:
        raise FileNotFoundError(f"Unable to decode video frames for {video_path}")

    video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
    sample_fps = nframes / max(ranged_total_frames, 1e-6) * source_fps
    video_metadata = {
        "fps": source_fps,
        "frames_indices": idx[: len(frames)],
        "total_num_frames": source_total_frames,
        "video_backend": "adaalloc-fallback-decoder",
    }
    return video, video_metadata, sample_fps


def _patch_qwen_video_readers(vision_process) -> None:
    """Fall back to our own decoder when a qwen-vl-utils reader fails on a video."""

    if getattr(vision_process, "_adaalloc_video_readers_patched", False):
        return

    original_decord = vision_process.VIDEO_READER_BACKENDS.get("decord")
    original_torchcodec = vision_process.VIDEO_READER_BACKENDS.get("torchcodec")

    def _wrap_reader(original_reader):
        def _reader(ele):
            try:
                return original_reader(ele)
            except Exception:
                return _read_qwen_video_with_fallback_decoder(ele, vision_process)

        return _reader

    if callable(original_decord):
        vision_process.VIDEO_READER_BACKENDS["decord"] = _wrap_reader(original_decord)
    if callable(original_torchcodec):
        vision_process.VIDEO_READER_BACKENDS["torchcodec"] = _wrap_reader(original_torchcodec)
    vision_process._adaalloc_video_readers_patched = True


def _configure_qwen_video_reader() -> None:
    """Make qwen-vl-utils choose a video reader that is healthy in this runtime.

    qwen-vl-utils detects torchcodec via importlib.find_spec(), which misses ABI mismatches. Configure the
    backend before import (decord when available, then torchcodec, then torchvision) and patch an
    already-loaded module.
    """

    from .video import _preload_conda_libwebp_for_decord, preferred_qwen_video_reader_backend

    _preload_conda_libwebp_for_decord()
    backend = os.environ.get("FORCE_QWENVL_VIDEO_READER")
    if not backend:
        backend = preferred_qwen_video_reader_backend()
        os.environ["FORCE_QWENVL_VIDEO_READER"] = backend

    vision_process = sys.modules.get("qwen_vl_utils.vision_process")
    if vision_process is not None:
        _patch_qwen_video_readers(vision_process)
        setattr(vision_process, "FORCE_QWENVL_VIDEO_READER", backend)
        get_backend = getattr(vision_process, "get_video_reader_backend", None)
        cache_clear = getattr(get_backend, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


def _import_process_vision_info():
    _configure_qwen_video_reader()
    from qwen_vl_utils import process_vision_info

    _configure_qwen_video_reader()
    return process_vision_info


class TransformersQwenReasoner:
    """Lazy-loaded transformers wrapper around Qwen3-VL.

    The planner and refiner calls run under request-local visual budgets; the uniform-frame answers run
    under ``answer_overrides``. All overrides are thread-local.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._processor = None
        self._model = None
        self._thread_local = threading.local()
        self._qwen_profile = resolve_qwen_profile(config.model.name)

    @property
    def last_request_metadata(self) -> dict[str, Any] | None:
        """Visual-token and preprocessing metadata of this thread's last model call."""

        return getattr(self._thread_local, "last_request_metadata", None)

    def _set_last_request_metadata(self, metadata: dict[str, Any]) -> None:
        self._thread_local.last_request_metadata = metadata

    @contextmanager
    def _thread_overrides(self, **values: Any):
        missing = object()
        previous = {name: getattr(self._thread_local, name, missing) for name in values}
        for name, value in values.items():
            setattr(self._thread_local, name, value)
        try:
            yield
        finally:
            for name, value in previous.items():
                if value is missing:
                    delattr(self._thread_local, name)
                else:
                    setattr(self._thread_local, name, value)

    @contextmanager
    def answer_overrides(
        self,
        *,
        prompt_style: str | None = None,
        video_path_fps: float | None = None,
        video_min_pixels: int | None = None,
        video_max_frames: int | None = None,
    ):
        """Answer with another prompt style and/or from the video file (uniform-frame answers).

        With ``video_path_fps``, an ``answer_question`` call given no decoded frames lets qwen-vl-utils sample
        the video file at that rate, with ``video_min_pixels`` per frame and at most ``video_max_frames``.
        """

        overrides: dict[str, Any] = {}
        if prompt_style is not None:
            if prompt_style not in ANSWER_PROMPT_STYLES:
                raise ValueError(f"Unsupported answer prompt style {prompt_style!r}; expected one of {ANSWER_PROMPT_STYLES}.")
            overrides["answer_prompt_style_override"] = prompt_style
        if video_path_fps is not None:
            overrides["video_path_fps_override"] = float(video_path_fps)
        if video_min_pixels is not None:
            overrides["video_min_pixels_override"] = int(video_min_pixels)
        if video_max_frames is not None:
            overrides["video_max_frames_override"] = int(video_max_frames)
        with self._thread_overrides(**overrides):
            yield

    @contextmanager
    def _overview_planner_visual_budget(self):
        """Apply the planner's pixel budget and frame cap to this thread's requests."""

        with self._thread_overrides(
            visual_budget_stage_override="overview",
            video_total_pixels_override=int(self.config.budget.total_pixels),
            video_max_frames_override=int(self.config.caps.planner),
        ):
            yield

    @contextmanager
    def _memory_refinement_visual_budget(self):
        """Apply the refiner's pixel budget and frame cap to this thread's requests."""

        with self._thread_overrides(
            visual_budget_stage_override="refinement",
            video_total_pixels_override=int(self.config.budget.total_pixels),
            video_max_frames_override=int(self.config.caps.refiner),
        ):
            yield

    def _effective_video_total_pixels(self) -> int:
        override = getattr(self._thread_local, "video_total_pixels_override", None)
        if override is not None:
            return int(override)
        return int(self.config.budget.total_pixels)

    def _effective_video_max_frames(self) -> int:
        override = getattr(self._thread_local, "video_max_frames_override", None)
        if override is not None:
            return int(override)
        return int(self.config.caps.local)

    def _effective_video_min_pixels(self) -> int:
        override = getattr(self._thread_local, "video_min_pixels_override", None)
        if override is not None:
            return int(override)
        return int(self.config.video.min_pixels)

    def _answer_prompt_style(self) -> str:
        return getattr(self._thread_local, "answer_prompt_style_override", None) or "grounded_structured_mcq"

    def _adaptive_raw_video_total_pixels(self) -> int:
        multiplier = float(self.config.video.raw_pixels_multiplier)
        return max(
            self._qwen_profile.min_frame_pixels,
            int(round(self._effective_video_total_pixels() * multiplier)),
        )

    @staticmethod
    def _ensure_writable_frame(frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8 or not frame.flags.writeable or not frame.flags.c_contiguous:
            return np.array(frame, dtype=np.uint8, copy=True, order="C")
        return frame

    def _ensure_writable_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return [self._ensure_writable_frame(frame) for frame in frames]

    @staticmethod
    def _infer_sample_fps(observed_frames: list[FrameReference], raw_frames: list[np.ndarray]) -> float | None:
        if not raw_frames or len(raw_frames) < 2 or len(observed_frames) != len(raw_frames):
            return None

        timestamps = [frame.timestamp_sec for frame in observed_frames]
        diffs = [curr - prev for prev, curr in zip(timestamps, timestamps[1:]) if curr > prev]
        if not diffs:
            return None

        mean_delta = sum(diffs) / len(diffs)
        if mean_delta <= 0:
            return None

        max_relative_deviation = max(abs(delta - mean_delta) / mean_delta for delta in diffs)
        if max_relative_deviation > 0.20:
            return None

        sample_fps = 1.0 / mean_delta
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            return None
        return float(sample_fps)

    @staticmethod
    def _build_video_metadata(
        sample: QuestionSample | None,
        observed_frames: list[FrameReference],
        raw_frames: list[np.ndarray],
    ) -> dict[str, object] | None:
        if not raw_frames or not observed_frames or len(observed_frames) != len(raw_frames):
            return None

        source_metadata = sample.video_metadata if sample is not None else None
        source_fps = source_metadata.fps if source_metadata is not None else None
        if source_fps is None and sample is not None and sample.video_path:
            try:
                from .video import probe_video_metadata

                probed_metadata = probe_video_metadata(sample.video_path, fallback_fps=None)
                if probed_metadata.fps is not None:
                    source_metadata = probed_metadata
                    source_fps = probed_metadata.fps
            except Exception:
                # Never fabricate full-video metadata from sparse frames; the caller falls back to an fps hint.
                pass
        frame_indices = [frame.frame_index for frame in observed_frames]
        if source_fps is None or any(index < 0 for index in frame_indices):
            return None

        total_num_frames = source_metadata.total_frames if source_metadata is not None else None
        if total_num_frames is None:
            total_num_frames = max(frame_indices) + 1 if frame_indices else len(raw_frames)

        height, width = raw_frames[0].shape[:2]
        duration_sec = source_metadata.duration_sec if source_metadata is not None else None
        if duration_sec is None and source_fps:
            duration_sec = total_num_frames / source_fps

        return {
            "fps": float(source_fps),
            "frames_indices": frame_indices,
            "total_num_frames": int(total_num_frames),
            "height": int(height),
            "width": int(width),
            "duration": float(duration_sec) if duration_sec is not None else None,
            "video_backend": "adaalloc-predecoded-frames",
        }

    def _should_use_video_path_preprocessing(
        self,
        sample: QuestionSample | None,
        raw_frames: list[np.ndarray] | VisualEvidenceInput | None,
    ) -> bool:
        return bool(
            sample is not None
            and not raw_frames
            and getattr(self._thread_local, "video_path_fps_override", None) is not None
        )

    def _preprocess_raw_frames_for_qwen(
        self,
        raw_frames: list[np.ndarray] | VisualEvidenceInput | None,
        observed_frames: list[FrameReference] | None,
    ) -> tuple[list[np.ndarray] | VisualEvidenceInput | None, list[FrameReference]]:
        if isinstance(raw_frames, VisualEvidenceInput):
            raw_frames.frames = self._ensure_writable_frames(list(raw_frames.frames))
            for stream in raw_frames.video_streams:
                stream.frames = self._ensure_writable_frames(list(stream.frames))
            return raw_frames, list(observed_frames or [])

        frames = self._ensure_writable_frames(list(raw_frames or []))
        frame_refs = list(observed_frames or [])
        if not frames:
            return raw_frames, frame_refs

        max_frames = self._effective_video_max_frames()
        if len(frames) > max_frames:
            selected_positions = np.round(np.linspace(0, len(frames) - 1, num=max_frames)).astype(int)
            frames = [frames[index] for index in selected_positions]
            if frame_refs:
                frame_refs = [frame_refs[index] for index in selected_positions]

        height, width = frames[0].shape[:2]
        max_total_pixels = int(self._adaptive_raw_video_total_pixels())
        max_total_pixels = min(max_total_pixels, int(self.config.video.max_pixels) * len(frames))
        min_total_pixels = max(self._qwen_profile.min_frame_pixels, self._effective_video_min_pixels() * len(frames))
        temporal_factor = self._qwen_profile.temporal_patch_size if len(frames) >= self._qwen_profile.temporal_patch_size else 1
        try:
            resized_height, resized_width = resize_size_for_profile(
                self._qwen_profile,
                frame_count=len(frames),
                height=height,
                width=width,
                temporal_factor=temporal_factor,
                min_total_pixels=min_total_pixels,
                max_total_pixels=max_total_pixels,
            )
        except Exception:
            return frames, frame_refs
        if resized_height == height and resized_width == width:
            return frames, frame_refs

        resized_frames: list[np.ndarray] = []
        for frame in frames:
            image = Image.fromarray(frame.astype(np.uint8, copy=False))
            resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
            resized_frames.append(np.array(resized, dtype=np.uint8, copy=True))
        return resized_frames, frame_refs

    def _qwen_grid_size_with_visual_token_floor(
        self,
        *,
        source_height: int,
        source_width: int,
        source_frame_count: int,
        padded_frame_count: int,
        min_visual_tokens_per_source_frame: int,
        max_total_pixels: int,
    ) -> tuple[int, int] | None:
        """Find an aspect-aware discrete grid that satisfies a token floor.

        Qwen's native smart resize is conservative and can round just below ``min_pixels``, which breaks a
        strict per-source-frame floor for one- and three-frame clips. Search the small discrete spatial grid
        directly and return the closest-aspect feasible size.
        """

        profile = self._qwen_profile
        source_frame_count = max(1, int(source_frame_count))
        padded_frame_count = max(source_frame_count, int(padded_frame_count))
        temporal_grid = math.ceil(padded_frame_count / profile.temporal_patch_size)
        required_total_tokens = source_frame_count * int(min_visual_tokens_per_source_frame)
        required_spatial_cells = math.ceil(required_total_tokens / temporal_grid)
        max_spatial_cells = int(max_total_pixels) // (padded_frame_count * profile.min_frame_pixels)
        if required_spatial_cells <= 0 or max_spatial_cells < required_spatial_cells:
            return None

        source_aspect = float(source_width) / float(source_height)
        candidates: list[tuple[float, float, int, int, int]] = []
        for grid_height in range(1, max_spatial_cells + 1):
            min_grid_width = math.ceil(required_spatial_cells / grid_height)
            max_grid_width = max_spatial_cells // grid_height
            if min_grid_width > max_grid_width:
                continue
            ideal_grid_width = source_aspect * grid_height
            candidate_widths = {
                min_grid_width,
                max_grid_width,
                max(min_grid_width, min(max_grid_width, int(math.floor(ideal_grid_width)))),
                max(min_grid_width, min(max_grid_width, int(math.ceil(ideal_grid_width)))),
            }
            for grid_width in candidate_widths:
                spatial_cells = grid_height * grid_width
                if not (required_spatial_cells <= spatial_cells <= max_spatial_cells):
                    continue
                candidate_aspect = float(grid_width) / float(grid_height)
                aspect_error = abs(math.log(candidate_aspect / source_aspect))
                token_overage = float(spatial_cells - required_spatial_cells) / float(required_spatial_cells)
                candidates.append(
                    (aspect_error + 0.05 * token_overage, token_overage, spatial_cells, grid_height, grid_width)
                )

        if not candidates:
            return None
        _, _, _, grid_height, grid_width = min(candidates)
        return grid_height * profile.spatial_factor, grid_width * profile.spatial_factor

    def _resize_frames_to_total_pixels(
        self,
        frames: list[np.ndarray],
        *,
        max_total_pixels: int,
        account_for_temporal_padding: bool = False,
        min_visual_tokens_per_source_frame: int | None = None,
    ) -> list[np.ndarray]:
        profile = self._qwen_profile
        if not frames:
            return []

        max_total_pixels = max(profile.min_frame_pixels, int(max_total_pixels))
        height, width = frames[0].shape[:2]
        budget_frame_count = len(frames)
        if account_for_temporal_padding:
            temporal_size = profile.temporal_patch_size
            budget_frame_count = math.ceil(len(frames) / temporal_size) * temporal_size
        min_total_pixels = min(
            max_total_pixels,
            max(profile.min_frame_pixels, profile.min_frame_pixels * budget_frame_count),
        )
        if min_visual_tokens_per_source_frame is not None:
            requested_min_total_pixels = (
                len(frames)
                * int(min_visual_tokens_per_source_frame)
                * profile.min_frame_pixels
                * profile.temporal_patch_size
            )
            min_total_pixels = min(max_total_pixels, max(min_total_pixels, requested_min_total_pixels))
        resized_height, resized_width = resize_size_for_profile(
            profile,
            frame_count=budget_frame_count,
            height=height,
            width=width,
            temporal_factor=profile.temporal_patch_size if account_for_temporal_padding else 1,
            min_total_pixels=min_total_pixels,
            max_total_pixels=max_total_pixels,
        )
        if min_visual_tokens_per_source_frame is not None:
            temporal_size = profile.temporal_patch_size
            padded_frame_count = math.ceil(budget_frame_count / temporal_size) * temporal_size
            realized_tokens = visual_tokens_from_grid_row(
                [
                    padded_frame_count // temporal_size,
                    resized_height // profile.patch_size,
                    resized_width // profile.patch_size,
                ],
                profile,
            )
            required_tokens = len(frames) * int(min_visual_tokens_per_source_frame)
            realized_total_pixels = padded_frame_count * resized_height * resized_width
            if realized_tokens < required_tokens or realized_total_pixels > max_total_pixels:
                floor_size = self._qwen_grid_size_with_visual_token_floor(
                    source_height=height,
                    source_width=width,
                    source_frame_count=len(frames),
                    padded_frame_count=padded_frame_count,
                    min_visual_tokens_per_source_frame=int(min_visual_tokens_per_source_frame),
                    max_total_pixels=max_total_pixels,
                )
                if floor_size is not None:
                    resized_height, resized_width = floor_size
        if resized_height == height and resized_width == width:
            return self._ensure_writable_frames(frames)

        resized_frames: list[np.ndarray] = []
        for frame in frames:
            image = Image.fromarray(frame.astype(np.uint8, copy=False))
            resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
            resized_frames.append(np.array(resized, dtype=np.uint8, copy=True))
        return resized_frames

    def _qwen_visual_token_budget(self) -> int:
        return max(1, self._effective_video_total_pixels() // self._qwen_profile.min_frame_pixels)

    def _prepared_video_visual_tokens(self, video: Any) -> int | None:
        shape = getattr(video, "shape", None)
        if shape is None or len(shape) != 4:
            return None
        frame_count, _, height, width = [int(value) for value in shape]
        temporal_grid = math.ceil(frame_count / self._qwen_profile.temporal_patch_size)
        height_grid = height // self._qwen_profile.patch_size
        width_grid = width // self._qwen_profile.patch_size
        if min(temporal_grid, height_grid, width_grid) <= 0:
            return None
        return visual_tokens_from_grid_row([temporal_grid, height_grid, width_grid], self._qwen_profile)

    def _cap_prepared_qwen_videos_to_visual_token_budget(
        self,
        videos: list[Any] | None,
    ) -> tuple[list[Any] | None, dict[str, Any]]:
        """Apply an exact post-rounding cap to qwen-vl-utils video tensors."""

        profile = self._qwen_profile
        if not videos:
            return videos, {}

        token_estimates = [self._prepared_video_visual_tokens(video) for video in videos]
        if any(estimate is None for estimate in token_estimates):
            return videos, {
                "qwen_visual_token_budget": self._qwen_visual_token_budget(),
                "qwen_post_resize_token_cap_applied": False,
                "qwen_post_resize_token_cap_skipped": "unsupported_video_shape",
            }

        estimates = [int(estimate) for estimate in token_estimates if estimate is not None]
        token_budget = self._qwen_visual_token_budget()
        total_before = sum(estimates)
        metadata: dict[str, Any] = {
            "qwen_visual_token_budget": token_budget,
            "qwen_preprocessor_visual_token_estimate_before_cap": total_before,
            "qwen_post_resize_token_cap_applied": False,
        }
        if total_before <= token_budget:
            metadata["qwen_preprocessor_visual_token_estimate_after_cap"] = total_before
            return videos, metadata

        import torch
        import torch.nn.functional as torch_functional

        resized_videos: list[Any] = []
        remaining_budget = token_budget
        shape_changes: list[dict[str, Any]] = []
        for video_index, (video, estimate) in enumerate(zip(videos, estimates)):
            if not isinstance(video, torch.Tensor):
                return videos, {**metadata, "qwen_post_resize_token_cap_skipped": "non_tensor_video"}

            streams_left = len(videos) - video_index
            if streams_left == 1:
                stream_budget = remaining_budget
            else:
                proportional = int(token_budget * estimate / max(total_before, 1))
                stream_budget = min(remaining_budget - (streams_left - 1), max(1, proportional))
            stream_budget = max(1, stream_budget)

            frame_count, _, height, width = [int(value) for value in video.shape]
            temporal_size = profile.temporal_patch_size
            padded_frame_count = math.ceil(frame_count / temporal_size) * temporal_size
            max_total_pixels = stream_budget * profile.min_frame_pixels * temporal_size
            min_total_pixels = padded_frame_count * profile.min_frame_pixels
            resized_height, resized_width = resize_size_for_profile(
                profile,
                frame_count=padded_frame_count,
                height=height,
                width=width,
                temporal_factor=temporal_size,
                min_total_pixels=min(min_total_pixels, max_total_pixels),
                max_total_pixels=max_total_pixels,
            )

            # Guard against discrete spatial rounding near the limit.
            resized_estimate = visual_tokens_from_grid_row(
                [
                    padded_frame_count // temporal_size,
                    resized_height // profile.patch_size,
                    resized_width // profile.patch_size,
                ],
                profile,
            )
            retry_pixels = max_total_pixels
            while resized_estimate > stream_budget:
                retry_pixels = max(min_total_pixels, int(retry_pixels * stream_budget / resized_estimate * 0.98))
                next_height, next_width = resize_size_for_profile(
                    profile,
                    frame_count=padded_frame_count,
                    height=height,
                    width=width,
                    temporal_factor=temporal_size,
                    min_total_pixels=min(min_total_pixels, retry_pixels),
                    max_total_pixels=retry_pixels,
                )
                if (next_height, next_width) == (resized_height, resized_width):
                    break
                resized_height, resized_width = next_height, next_width
                resized_estimate = visual_tokens_from_grid_row(
                    [
                        padded_frame_count // temporal_size,
                        resized_height // profile.patch_size,
                        resized_width // profile.patch_size,
                    ],
                    profile,
                )

            resized_video = video
            if (resized_height, resized_width) != (height, width):
                original_dtype = video.dtype
                resized_video = torch_functional.interpolate(
                    video.float(),
                    size=(resized_height, resized_width),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                if original_dtype != torch.float32:
                    resized_video = resized_video.to(dtype=original_dtype)
            resized_videos.append(resized_video)
            shape_changes.append(
                {
                    "stream_index": video_index,
                    "before": [frame_count, height, width],
                    "after": [frame_count, resized_height, resized_width],
                    "visual_tokens_before": estimate,
                    "visual_tokens_after": resized_estimate,
                    "visual_token_budget": stream_budget,
                }
            )
            remaining_budget = max(0, remaining_budget - resized_estimate)
        total_after = sum(int(self._prepared_video_visual_tokens(video) or 0) for video in resized_videos)
        if total_after > token_budget:
            raise ValueError(
                "Unable to fit prepared Qwen video inside the configured visual-token "
                f"budget: {total_after} > {token_budget}."
            )
        metadata.update(
            {
                "qwen_post_resize_token_cap_applied": True,
                "qwen_preprocessor_visual_token_estimate_after_cap": total_after,
                "qwen_post_resize_shape_changes": shape_changes,
            }
        )
        return resized_videos, metadata

    @staticmethod
    def _pad_frame_indices_for_temporal_patches(
        frame_indices: list[int],
        frame_count: int,
        *,
        temporal_patch_size: int = 2,
    ) -> list[int]:
        if not frame_indices:
            frame_indices = list(range(frame_count))
        padded = list(frame_indices)
        target_count = max(frame_count, len(padded))
        if target_count <= 0:
            return padded
        padded.extend([padded[-1]] * max(0, target_count - len(padded)))
        remainder = len(padded) % temporal_patch_size
        if remainder:
            padded.extend([padded[-1]] * (temporal_patch_size - remainder))
        return padded

    @staticmethod
    def _qwen_video_metadata(metadata: dict[str, Any], frame_count: int) -> dict[str, Any]:
        frame_indices = [int(index) for index in metadata.get("frames_indices", [])]
        inferred_total_frames = max(frame_indices, default=frame_count - 1) + 1
        return {
            "total_num_frames": int(metadata.get("total_num_frames") or inferred_total_frames),
            "fps": float(metadata["fps"]) if metadata.get("fps") is not None else None,
            "width": int(metadata["width"]) if metadata.get("width") is not None else None,
            "height": int(metadata["height"]) if metadata.get("height") is not None else None,
            "duration": float(metadata["duration"]) if metadata.get("duration") is not None else None,
            "video_backend": metadata.get("video_backend"),
            "frames_indices": TransformersQwenReasoner._pad_frame_indices_for_temporal_patches(frame_indices, frame_count),
        }

    def _prepare_hybrid_video_streams(
        self,
        raw_frames: VisualEvidenceInput,
        sample: QuestionSample | None,
    ) -> tuple[list[list[np.ndarray]], list[dict[str, Any]], list[float], list[dict[str, str]], dict[str, Any]]:
        """Resize each evidence stream to its pixel budget and build its label and timing metadata."""

        streams = list(raw_frames.video_streams or [])
        if not streams:
            raise ValueError("video_streams evidence requires at least one stream")

        videos: list[list[np.ndarray]] = []
        video_metadata: list[dict[str, Any]] = []
        video_fps: list[float] = []
        content_prefix: list[dict[str, str]] = []
        stream_summaries: list[dict[str, Any]] = []
        for stream_index, stream in enumerate(streams):
            stream_metadata = dict(stream.metadata)
            min_visual_tokens_per_frame = stream_metadata.get("qwen_min_visual_tokens_per_frame")
            if min_visual_tokens_per_frame is None and stream.kind == "segment":
                min_visual_tokens_per_frame = stream_metadata.get("qwen_local_min_visual_tokens_per_frame")
            if min_visual_tokens_per_frame is None and stream.kind == "overview":
                min_visual_tokens_per_frame = stream_metadata.get("qwen_overview_min_visual_tokens_per_frame")
            resized_frames = self._resize_frames_to_total_pixels(
                list(stream.frames),
                max_total_pixels=stream.total_pixels or self._adaptive_raw_video_total_pixels(),
                account_for_temporal_padding=True,
                min_visual_tokens_per_source_frame=(
                    int(min_visual_tokens_per_frame) if min_visual_tokens_per_frame is not None else None
                ),
            )
            if not resized_frames:
                continue
            videos.append(resized_frames)
            metadata = stream_metadata
            frame_refs = list(stream.frame_references)
            if "fps" not in metadata and sample is not None and sample.video_metadata.fps is not None:
                metadata["fps"] = float(sample.video_metadata.fps)
            if "frames_indices" not in metadata:
                metadata["frames_indices"] = [frame.frame_index for frame in frame_refs]
            if "timestamps_sec" not in metadata:
                metadata["timestamps_sec"] = [float(frame.timestamp_sec) for frame in frame_refs]
            if "total_num_frames" not in metadata and sample is not None:
                total_frames = sample.video_metadata.total_frames
                if total_frames is None and metadata.get("frames_indices"):
                    total_frames = max(int(index) for index in metadata["frames_indices"]) + 1
                if total_frames is not None:
                    metadata["total_num_frames"] = int(total_frames)
            if "duration" not in metadata and sample is not None and sample.video_metadata.duration_sec is not None:
                metadata["duration"] = float(sample.video_metadata.duration_sec)
            metadata.setdefault("video_backend", "adaalloc-hybrid-video-streams")
            metadata.setdefault("stream_index", stream_index)
            metadata.setdefault("stream_label", stream.label)
            metadata.setdefault("stream_kind", stream.kind)
            height, width = resized_frames[0].shape[:2]
            metadata["height"] = int(height)
            metadata["width"] = int(width)
            video_metadata.append(self._qwen_video_metadata(metadata, len(resized_frames)))

            timestamps = [float(frame.timestamp_sec) for frame in frame_refs[: len(resized_frames)]]
            positive_deltas = [
                current - previous for previous, current in zip(timestamps, timestamps[1:]) if current > previous
            ]
            if positive_deltas:
                effective_fps = len(positive_deltas) / sum(positive_deltas)
                timing_policy = "mean_positive_timestamp_delta"
            else:
                source_fps = metadata.get("fps")
                effective_fps = float(source_fps) if source_fps is not None else 1.0
                timing_policy = "source_fps_single_or_duplicate_frame"
            if not math.isfinite(effective_fps) or effective_fps <= 0:
                effective_fps = 1.0
                timing_policy = "safe_default"
            video_fps.append(float(effective_fps))
            source_range = f"{min(timestamps):.2f}s..{max(timestamps):.2f}s" if timestamps else "unknown"
            label_text = (
                f"Video stream {len(videos)} — {stream.label}: separately encoded {stream.kind} "
                f"evidence from source span {source_range}, containing {len(resized_frames)} visual frames "
                f"at effective sampling rate {effective_fps:.6g} FPS. "
                "This clip starts at model-relative time zero; use the per-stream source-frame and "
                "absolute-timestamp mapping in the question text for original-timeline reasoning."
            )
            content_prefix.extend([{"type": "text", "text": label_text}, {"type": "video"}])
            seconds_per_grid_ts = self._qwen_profile.temporal_patch_size / effective_fps
            stream_summaries.append(
                {
                    "label": stream.label,
                    "kind": stream.kind,
                    "frame_indices": list(metadata.get("frames_indices", [])),
                    "timestamps_sec": list(metadata.get("timestamps_sec", [])),
                    "total_pixels": int(stream.total_pixels),
                    "effective_sampling_fps": float(effective_fps),
                    "seconds_per_grid_ts": float(seconds_per_grid_ts),
                    "timing_policy": timing_policy,
                    "source_time_range_sec": [min(timestamps), max(timestamps)] if timestamps else None,
                    "requested_min_visual_tokens_per_source_frame": (
                        int(min_visual_tokens_per_frame) if min_visual_tokens_per_frame is not None else None
                    ),
                }
            )

        if not videos:
            raise ValueError("video_streams evidence has no non-empty streams")
        request_metadata = {
            "hybrid_visual_input_mode": "video_streams",
            "video_stream_count": len(videos),
            "video_streams": stream_summaries,
            "video_stream_effective_fps": list(video_fps),
            "video_stream_seconds_per_grid_ts": [
                float(self._qwen_profile.temporal_patch_size / fps) for fps in video_fps
            ],
        }
        return videos, video_metadata, video_fps, content_prefix, request_metadata

    def _extract_visual_token_metadata(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        video_grid_thw = model_inputs.get("video_grid_thw")
        if video_grid_thw is None:
            return {}

        if hasattr(video_grid_thw, "detach"):
            grid_rows = video_grid_thw.detach().cpu().tolist()
        else:
            grid_rows = list(video_grid_thw)

        normalized_rows: list[list[int]] = []
        visual_token_count = 0
        padded_frame_count = 0
        resized_height = None
        resized_width = None
        for row in grid_rows:
            t, h, w = [int(value) for value in row]
            normalized_rows.append([t, h, w])
            visual_token_count += visual_tokens_from_grid_row([t, h, w], self._qwen_profile)
            padded_frame_count += t * self._qwen_profile.temporal_patch_size
            if resized_height is None:
                resized_height = h * self._qwen_profile.patch_size
                resized_width = w * self._qwen_profile.patch_size

        return {
            "visual_token_count": int(visual_token_count),
            "visual_token_budget": self._qwen_visual_token_budget(),
            "visual_token_budget_exceeded": bool(visual_token_count > self._qwen_visual_token_budget()),
            "video_grid_thw": normalized_rows,
            "visual_stream_count": len(normalized_rows),
            "visual_tokens_by_stream": [
                int(visual_tokens_from_grid_row(row, self._qwen_profile)) for row in normalized_rows
            ],
            "padded_frame_count": int(padded_frame_count),
            "resized_height": resized_height,
            "resized_width": resized_width,
        }

    @staticmethod
    def _annotate_video_stream_token_density(metadata: dict[str, Any]) -> None:
        """Attach processor-realized token density to each evidence stream."""

        streams = metadata.get("video_streams")
        tokens_by_stream = metadata.get("visual_tokens_by_stream")
        if not isinstance(streams, list) or not isinstance(tokens_by_stream, list):
            return

        totals = {
            kind: {"tokens": 0, "source_frames": 0, "checks": [], "requested": []} for kind in ("segment", "overview")
        }
        for stream_index, stream in enumerate(streams):
            if not isinstance(stream, dict) or stream_index >= len(tokens_by_stream):
                continue
            visual_tokens = int(tokens_by_stream[stream_index])
            frame_indices = stream.get("frame_indices")
            source_frame_count = len(frame_indices) if isinstance(frame_indices, list) else 0
            realized_density = float(visual_tokens) / float(source_frame_count) if source_frame_count > 0 else None
            stream["visual_token_count"] = visual_tokens
            stream["source_frame_count"] = source_frame_count
            stream["realized_visual_tokens_per_source_frame"] = realized_density
            kind_totals = totals.get(stream.get("kind"))
            requested_density = stream.get("requested_min_visual_tokens_per_source_frame")
            if requested_density is not None and realized_density is not None:
                requested_density = int(requested_density)
                density_satisfied = realized_density + 1e-9 >= requested_density
                stream["requested_min_visual_tokens_per_source_frame_satisfied"] = density_satisfied
                if kind_totals is not None:
                    kind_totals["checks"].append(density_satisfied)
                    kind_totals["requested"].append(requested_density)
            if kind_totals is not None:
                kind_totals["tokens"] += visual_tokens
                kind_totals["source_frames"] += source_frame_count

        for kind, prefix in (("overview", "qwen_overview"), ("segment", "qwen_local")):
            kind_totals = totals[kind]
            if kind_totals["source_frames"] <= 0:
                continue
            metadata[f"{prefix}_visual_token_count"] = int(kind_totals["tokens"])
            metadata[f"{prefix}_source_frame_count"] = int(kind_totals["source_frames"])
            metadata[f"{prefix}_realized_visual_tokens_per_source_frame"] = float(kind_totals["tokens"]) / float(
                kind_totals["source_frames"]
            )
            if kind_totals["requested"]:
                metadata[f"{prefix}_requested_min_visual_tokens_per_source_frame"] = min(kind_totals["requested"])
                metadata[f"{prefix}_min_visual_tokens_per_source_frame_satisfied"] = bool(kind_totals["checks"]) and all(
                    kind_totals["checks"]
                )

    def _build_model_inputs(
        self,
        prompt: str,
        *,
        raw_frames: list[np.ndarray] | VisualEvidenceInput | None,
        observed_frames: list[FrameReference] | None,
        sample: QuestionSample | None,
        processor,
    ) -> dict[str, Any]:
        """Build processor inputs from the video file, from evidence streams, or from one raw frame list."""

        if self._should_use_video_path_preprocessing(sample, raw_frames):
            assert sample is not None
            images, videos, video_metadatas, video_kwargs, video_path_metadata = self._prepare_inputs_from_video_path(
                sample,
                max_frames=len(observed_frames or []),
                processor=processor,
            )
            model_inputs = self._build_processor_inputs_from_video_path(
                prompt,
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                video_kwargs=video_kwargs,
                processor=processor,
            )
            model_inputs[_REQUEST_METADATA_KEY] = video_path_metadata
            return model_inputs

        raw_frames, observed_frames = self._preprocess_raw_frames_for_qwen(raw_frames, observed_frames)
        if isinstance(raw_frames, VisualEvidenceInput):
            videos, video_metadata, video_fps, content_prefix, request_metadata = self._prepare_hybrid_video_streams(
                raw_frames, sample
            )
            rendered = processor.apply_chat_template(
                [{"role": "user", "content": [*content_prefix, {"type": "text", "text": prompt}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = processor(
                text=[rendered],
                videos=videos,
                video_metadata=video_metadata,
                fps=video_fps,
                return_tensors="pt",
                do_sample_frames=False,
                do_resize=False,
            )
            model_inputs[_REQUEST_METADATA_KEY] = request_metadata
            return model_inputs

        frames = list(raw_frames or [])
        if frames:
            content = [{"type": "video"}, {"type": "text", "text": prompt}]
        else:
            content = [{"type": "text", "text": prompt}]
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        processor_kwargs: dict[str, object] = {"text": [rendered], "return_tensors": "pt"}
        if frames:
            processor_kwargs["videos"] = [frames]
            processor_kwargs["do_sample_frames"] = False
            processor_kwargs["do_resize"] = False
            video_metadata = self._build_video_metadata(sample, observed_frames or [], frames)
            if video_metadata is not None:
                processor_kwargs["video_metadata"] = video_metadata
            else:
                inferred_sample_fps = self._infer_sample_fps(observed_frames or [], frames)
                if inferred_sample_fps is not None:
                    processor_kwargs["fps"] = inferred_sample_fps
        return processor(**processor_kwargs)

    def _prepare_inputs_from_video_path(
        self,
        sample: QuestionSample,
        *,
        max_frames: int,
        processor,
    ) -> tuple[list[Any] | None, list[Any] | None, list[dict[str, Any]] | None, dict[str, Any], dict[str, Any]]:
        """Let qwen-vl-utils sample and resize the video file, then cap the result to the token budget."""

        process_vision_info = _import_process_vision_info()

        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", 16)
        min_frames = int(self.config.video.min_frames)
        configured_max_frames = self._effective_video_max_frames()
        effective_max_frames = max(max_frames, min_frames)
        effective_max_frames = max(min_frames, min(effective_max_frames, configured_max_frames))
        video_min_pixels = self._effective_video_min_pixels()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": sample.video_path,
                        "fps": self._thread_local.video_path_fps_override,
                        "min_frames": min_frames,
                        "max_frames": effective_max_frames,
                        "min_pixels": video_min_pixels,
                        "max_pixels": self.config.video.max_pixels,
                        "total_pixels": self._effective_video_total_pixels(),
                    },
                    {"type": "text", "text": "Prepare video frames for question answering."},
                ],
            }
        ]
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            videos, token_cap_metadata = self._cap_prepared_qwen_videos_to_visual_token_budget(videos)
        else:
            video_metadatas = None
            token_cap_metadata = {}

        metadata: dict[str, Any] = {
            "qwen_video_path_preprocessing": True,
            "requested_frame_count": 0,
            "reasoning_frame_count": 0,
            "decoded_frame_indices": [],
            "decoded_frame_count": 0,
            "qwen_video_path_requested_max_frames": int(max_frames),
            "qwen_video_path_configured_max_frames": int(configured_max_frames),
            "qwen_video_path_effective_max_frames": int(effective_max_frames),
            "qwen_video_path_effective_min_pixels": int(video_min_pixels),
            "qwen_video_path_effective_max_pixels": int(self.config.video.max_pixels),
            "qwen_video_path_effective_total_pixels": int(self._effective_video_total_pixels()),
            **token_cap_metadata,
        }
        if video_metadatas:
            video_metadata = video_metadatas[0]
            frame_indices = [int(index) for index in video_metadata.get("frames_indices", [])]
            metadata.update(
                {
                    "requested_frame_count": len(frame_indices),
                    "reasoning_frame_count": len(frame_indices),
                    "decoded_frame_indices": frame_indices,
                    "decoded_frame_count": len(frame_indices),
                    "qwen_video_path_actual_frame_count": len(frame_indices),
                    "qwen_video_path_actual_frame_indices": frame_indices,
                    "qwen_video_metadata": dict(video_metadata),
                }
            )
        return images, videos, video_metadatas, video_kwargs, metadata

    @staticmethod
    def _build_processor_inputs_from_video_path(
        prompt: str,
        *,
        images,
        videos,
        video_metadatas,
        video_kwargs,
        processor,
    ) -> dict[str, Any]:
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return processor(
            text=[rendered],
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )

    def _load_processor(self) -> None:
        if self._processor is not None:
            return

        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.config.model.name, trust_remote_code=True)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        self._load_processor()

        import torch
        from transformers import AutoModelForImageTextToText

        device = self.config.model.device
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": getattr(torch, self.config.model.dtype),
        }
        if device.startswith("cuda"):
            try:
                import accelerate  # noqa: F401
            except ImportError:
                pass
            else:
                model_kwargs["device_map"] = device

        model = AutoModelForImageTextToText.from_pretrained(self.config.model.name, **model_kwargs)
        if "device_map" not in model_kwargs:
            model = model.to(device)
        model.eval()
        self._model = model

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        self._load_processor()
        self._load_model()

    def _generation_kwargs(self) -> dict[str, object]:
        max_new_tokens_override = getattr(self._thread_local, "max_new_tokens_override", None)
        return {
            "max_new_tokens": (
                int(max_new_tokens_override) if max_new_tokens_override is not None else self.config.model.max_new_tokens
            ),
            "repetition_penalty": 1.0,
            "do_sample": False,
        }

    def _complete_text(
        self,
        prompt: str,
        raw_frames: list[np.ndarray] | VisualEvidenceInput | None = None,
        observed_frames: list[FrameReference] | None = None,
        sample: QuestionSample | None = None,
    ) -> str:
        """Run one greedy generation and record its visual metadata as ``last_request_metadata``."""

        self._load()
        assert self._processor is not None
        assert self._model is not None
        self._set_last_request_metadata({})

        model_inputs = self._build_model_inputs(
            prompt,
            raw_frames=raw_frames,
            observed_frames=observed_frames,
            sample=sample,
            processor=self._processor,
        )
        request_metadata = model_inputs.pop(_REQUEST_METADATA_KEY, None)
        visual_metadata = dict(request_metadata) if isinstance(request_metadata, dict) else {}
        visual_metadata.update(self._extract_visual_token_metadata(model_inputs))
        self._annotate_video_stream_token_density(visual_metadata)
        visual_budget_stage = getattr(self._thread_local, "visual_budget_stage_override", None)
        visual_metadata.setdefault("qwen_visual_token_budget", self._qwen_visual_token_budget())
        visual_metadata.setdefault("qwen_video_total_pixels", self._effective_video_total_pixels())
        visual_metadata.setdefault("qwen_overview_planner_budget_override_active", visual_budget_stage == "overview")
        visual_metadata.setdefault("qwen_video_max_frames", self._effective_video_max_frames())
        visual_metadata.setdefault("qwen_overview_planner_frame_override_active", visual_budget_stage == "overview")
        visual_metadata.setdefault("qwen_refinement_budget_override_active", visual_budget_stage == "refinement")
        visual_metadata.setdefault("qwen_refinement_frame_override_active", visual_budget_stage == "refinement")

        device = self.config.model.device
        model_inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in model_inputs.items()}
        generated = self._model.generate(**model_inputs, **self._generation_kwargs())
        if "input_ids" in model_inputs and hasattr(generated, "shape") and generated.ndim == 2:
            generated = generated[:, model_inputs["input_ids"].shape[1] :]
        decoded = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if visual_metadata:
            self._set_last_request_metadata(visual_metadata)
        return decoded

    def _build_answer_prompt(
        self,
        sample: QuestionSample,
        retrieved_segments: list[RetrievedSegment],
        raw_frames: list[np.ndarray] | VisualEvidenceInput,
    ) -> str:
        if self._answer_prompt_style() == "simple_mcq":
            return build_simple_answer_prompt(sample)
        return build_grounded_structured_answer_prompt(
            sample,
            render_video_stream_evidence(raw_frames.video_streams),
            retrieved_segments,
            video_streams=True,
        )

    def _build_answer_retry_prompt(self, prompt: str, raw_output: str) -> str:
        if self._answer_prompt_style() == "simple_mcq":
            return (
                f"{prompt}\n\n"
                "Your previous answer was not in the required format.\n"
                "Respond again with ONLY the single option letter (for example: A).\n"
                "Do not include any explanation or extra text.\n\n"
                f"Previous raw response:\n{raw_output[:1200]}"
            )
        return (
            f"{prompt}\n\n"
            "Your previous answer could not be parsed into a valid final response.\n"
            "Respond again with ONLY one valid JSON object.\n"
            "Requirements:\n"
            "- selected_option must be exactly one of the option letters shown above.\n"
            "- selected_option_text must exactly match the chosen option text.\n"
            "- confidence must be a number between 0.0 and 1.0.\n"
            "- reasoning must briefly cite the visible evidence.\n"
            "Do not include markdown fences, commentary, or any text before/after the JSON.\n\n"
            f"Previous raw response:\n{raw_output[:1200]}"
        )

    @staticmethod
    def _build_overview_plan_retry_prompt(prompt: str, raw_output: str, parse_error: ValueError) -> str:
        return (
            f"{prompt}\n\n"
            "Your previous planning response was not valid under the required schema.\n"
            f"Parser error: {str(parse_error)[:500]}\n"
            "Respond again with ONLY one valid JSON object. Include rationale, budget_policy, "
            "and local_retrieval. Include temporal_proposals as a JSON list when proposals "
            "exist; it may be omitted or empty when there are none. Do not include markdown "
            "fences or commentary.\n\n"
            f"Previous raw response:\n{raw_output[:1600]}"
        )

    @staticmethod
    def _build_memory_refinement_retry_prompt(prompt: str, raw_output: str, parse_error: ValueError) -> str:
        return (
            f"{prompt}\n\n"
            "Your previous memory-refinement response was not valid under "
            "the required schema.\n"
            f"Parser error: {str(parse_error)[:500]}\n"
            "Respond again with ONLY one complete JSON object. Its first key "
            "must be actions, with exactly one entry per region listed "
            "above (keep, adjust, merge, or remove), followed by "
            "usable_local_evidence, uncertainty, and notes. Keep rationales "
            "short so the object is complete. Do not include markdown "
            "fences or commentary.\n\n"
            f"Previous raw response:\n{raw_output[:1600]}"
        )

    def generate_overview_plan(
        self,
        sample: QuestionSample,
        observed_frames: list[FrameReference],
        raw_frames: list[np.ndarray],
    ) -> AdaptiveOverviewPlan:
        """Plan the budget split and local queries from a sparse whole-video overview."""

        with self._overview_planner_visual_budget():
            return self._generate_overview_plan_with_effective_budget(sample, observed_frames, raw_frames)

    def _generate_overview_plan_with_effective_budget(
        self,
        sample: QuestionSample,
        observed_frames: list[FrameReference],
        raw_frames: list[np.ndarray],
    ) -> AdaptiveOverviewPlan:
        prompt = build_overview_planning_prompt(sample, render_frame_evidence(observed_frames))
        attempt_prompt = prompt
        last_error: ValueError | None = None
        max_attempts = max(1, int(self.config.model.plan_parse_attempts))
        for attempt in range(max_attempts):
            raw_output = self._complete_text(
                attempt_prompt,
                raw_frames=raw_frames,
                observed_frames=observed_frames,
                sample=sample,
            )
            try:
                return parse_overview_plan(raw_output, duration_sec=sample.video_metadata.duration_sec)
            except ValueError as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    attempt_prompt = self._build_overview_plan_retry_prompt(prompt, raw_output, exc)
        assert last_error is not None
        return fallback_overview_plan(_fallback_initial_queries(sample), reason=f"parse_failure:{last_error}")

    def refine_memory_bank(
        self,
        sample: QuestionSample,
        memory_regions: list[Any],
        observed_frames: list[FrameReference],
        raw_frames: VisualEvidenceInput,
        retrieved_segments: list[RetrievedSegment] | None = None,
        *,
        refinement_context: dict[str, Any] | None = None,
    ) -> AdaptiveMemoryRefinement:
        """Review the candidate regions (one video stream each) with the v6 refinement prompt."""

        context = refinement_context or {}
        prompt = build_memory_refinement_prompt_v6(
            sample,
            memory_regions,
            render_video_stream_evidence(raw_frames.video_streams),
            retrieved_segments,
            plan=context.get("plan"),
            frame_counts_by_region=context.get("frame_counts_by_region"),
            context_clip_count=int(context.get("context_clip_count") or 0),
        )
        raw_outputs: list[str] = []

        def _record_attempts(attempts: int) -> None:
            request_metadata = self.last_request_metadata
            if isinstance(request_metadata, dict):
                request_metadata["adaptive_memory_refinement_parse_attempts"] = attempts
                request_metadata["adaptive_memory_refinement_raw_outputs"] = list(raw_outputs)

        attempt_prompt = prompt
        last_error: ValueError | None = None
        max_attempts = max(1, int(self.config.refiner.parse_attempts))
        with self._thread_overrides(max_new_tokens_override=int(self.config.refiner.max_new_tokens)):
            with self._memory_refinement_visual_budget():
                for attempt in range(max_attempts):
                    raw_output = self._complete_text(
                        attempt_prompt,
                        raw_frames=raw_frames,
                        observed_frames=observed_frames,
                        sample=sample,
                    )
                    raw_outputs.append(str(raw_output)[:_REFINEMENT_RAW_OUTPUT_CHARS])
                    try:
                        parsed = parse_memory_refinement_v6(raw_output)
                    except ValueError as exc:
                        last_error = exc
                        if attempt < max_attempts - 1:
                            attempt_prompt = self._build_memory_refinement_retry_prompt(prompt, raw_output, exc)
                        continue
                    _record_attempts(attempt + 1)
                    return parsed
        assert last_error is not None
        _record_attempts(max_attempts)
        return fallback_memory_refinement(reason=f"parse_failure:{last_error}")

    def answer_question(
        self,
        sample: QuestionSample,
        observed_frames: list[FrameReference],
        raw_frames: list[np.ndarray] | VisualEvidenceInput,
        retrieved_segments: list[RetrievedSegment],
    ) -> AnswerPrediction:
        """Answer the multiple-choice question, retrying unparsable replies before a deterministic fallback."""

        prompt = self._build_answer_prompt(sample, retrieved_segments, raw_frames)
        max_attempts = max(1, self.config.model.answer_parse_attempts)
        attempt_prompt = prompt
        raw_outputs: list[str] = []
        for attempt in range(max_attempts):
            raw_output = self._complete_text(
                attempt_prompt,
                raw_frames=raw_frames,
                observed_frames=observed_frames,
                sample=sample,
            )
            raw_outputs.append(raw_output)
            try:
                return parse_answer_prediction(sample, raw_output)
            except ValueError:
                if attempt == max_attempts - 1:
                    break
                attempt_prompt = self._build_answer_retry_prompt(prompt, raw_output)
        return fallback_answer_prediction(sample, raw_outputs)


def build_reasoner(config: Config) -> TransformersQwenReasoner:
    """Build the (lazily loaded) Qwen3-VL reasoner."""

    return TransformersQwenReasoner(config)
