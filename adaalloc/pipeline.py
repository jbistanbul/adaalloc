"""The AdaAlloc per-question pipeline.

The planner reads a sparse overview and picks an allocation policy and retrieval queries, UniversalVTG retrieves candidate moments into a memory bank, the refiner
checks every candidate region on its own clip, and the final answer sees an overview stream plus one
detail stream per region, splitting the visual-token budget as the policy says.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from .config import Config
from .data import (
    AnswerPrediction,
    FrameReference,
    QuestionResult,
    QuestionSample,
    RetrievedSegment,
    VideoEvidenceStream,
    VisualEvidenceInput,
)
from .memory import (
    AdaptiveAllocation,
    allocate_visual_budget,
    apply_refinement,
    build_memory_bank,
    candidate_frame_indices_by_region,
    is_order_or_count_question,
    parse_time_reference,
    regions_to_retrieved_segments,
    sample_candidate_frame_indices,
)
from .planning import (
    AdaptiveMemoryRefinement,
    AdaptiveOverviewPlan,
    AdaptiveTemporalProposal,
    fallback_memory_refinement,
    force_allocation_policy,
    option_restated_by_query,
)
from .qwen import resize_size_for_profile, resolve_qwen_profile, visual_tokens_from_grid_row
from .retrieval import sample_frames_from_retrieved_segments
from .video import (
    decode_frames_at_indices,
    probe_video_dimensions,
    probe_video_metadata,
    qwen_fps_frame_indices,
    resize_frames_to_total_pixels,
    uniform_frame_indices,
    video_metadata_complete,
)

if TYPE_CHECKING:
    from .retrieval import UniversalVTGBackend

# Smallest frame the local stream may be resized to: one 32x32 Qwen3-VL token cell.
MIN_FEASIBLE_LOCAL_FRAME_PIXELS = 32 * 32
SIMPLE_MCQ_PROMPT_STYLE = "simple_mcq"
VIDEO_STREAM_BACKEND = "adaalloc-video-streams"


# ----------------------------------------------------------------------------------------------------
# Sample preparation and frame helpers
# ----------------------------------------------------------------------------------------------------


def prepare_sample(config: Config, sample: QuestionSample) -> QuestionSample:
    """Fill any missing video metadata (frame count, fps, duration) by probing the file, in place."""

    metadata = sample.video_metadata
    if not video_metadata_complete(metadata):
        probed = probe_video_metadata(sample.video_path, fallback_fps=metadata.fps)
        if metadata.total_frames is None:
            metadata.total_frames = probed.total_frames
        if metadata.fps is None:
            metadata.fps = probed.fps
        if metadata.duration_sec is None:
            metadata.duration_sec = probed.duration_sec
        metadata.source = metadata.source if metadata.source != "unknown" else probed.source
    return sample


def _decode_selected_frames(sample: QuestionSample, indices: list[int]) -> list:
    return decode_frames_at_indices(sample.video_path, sorted(set(indices)))


def _build_frame_references(reasoning_indices: list[int], fps: float) -> list[FrameReference]:
    return [
        FrameReference(frame_index=frame_index, timestamp_sec=frame_index / fps)
        for frame_index in sorted(set(reasoning_indices))
    ]


def _with_evidence_metadata(
    refs: list[FrameReference],
    *,
    evidence_type: str,
    evidence_role: str,
    local_segments: list[RetrievedSegment] | None = None,
) -> list[FrameReference]:
    annotated: list[FrameReference] = []
    local_segment_lookup = [
        (
            index,
            float(segment.start_sec),
            float(segment.end_sec),
            {int(frame_index) for frame_index in segment.frame_indices},
        )
        for index, segment in enumerate(local_segments or [], start=1)
    ]
    for ref in refs:
        evidence_region = ""
        if local_segment_lookup:
            matching_regions = []
            for index, start_sec, end_sec, frame_indices in local_segment_lookup:
                in_time_span = start_sec <= float(ref.timestamp_sec) <= end_sec
                in_frame_indices = ref.frame_index in frame_indices
                if in_time_span or in_frame_indices:
                    matching_regions.append(f"local_{index}")
            evidence_region = ",".join(matching_regions)
        annotated.append(
            FrameReference(
                frame_index=ref.frame_index,
                timestamp_sec=ref.timestamp_sec,
                evidence_type=evidence_type,
                evidence_role=evidence_role,
                evidence_region=evidence_region,
            )
        )
    return annotated


def _prepare_reasoning_inputs(
    sample: QuestionSample,
    candidate_indices: list[int],
    *,
    fps: float,
) -> tuple[list[FrameReference], list, dict[str, Any]]:
    """Decode the given frames for a raw-frame request."""

    reasoning_indices = sorted(set(int(index) for index in candidate_indices))
    raw_frames = _decode_selected_frames(sample, reasoning_indices)
    frame_refs = _build_frame_references(reasoning_indices, fps)
    return frame_refs, raw_frames, {
        "requested_frame_count": len(reasoning_indices),
        "decoded_frame_indices": [frame.frame_index for frame in frame_refs],
        "decoded_frame_count": len(raw_frames),
    }


def _prepare_video_path_reasoning_inputs(
    candidate_indices: list[int],
    *,
    fps: float,
) -> tuple[list[FrameReference], list, dict[str, Any]]:
    """Reference the given frames for Qwen's native video reader (nothing is decoded here)."""

    reasoning_indices = sorted(set(int(index) for index in candidate_indices))
    return _build_frame_references(reasoning_indices, fps), [], {
        "requested_frame_count": len(reasoning_indices),
        "decoded_frame_indices": [],
        "decoded_frame_count": 0,
        "qwen_video_path_preprocessing": True,
    }


def _estimated_total_frame_pixels(frames: list[Any]) -> int:
    total_pixels = 0
    for frame in frames:
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) < 2:
            continue
        total_pixels += int(shape[0]) * int(shape[1])
    return total_pixels


def _augment_qwen_cache_metadata(frame_metadata: dict[str, Any], reasoning_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Add the visual tokens and grid the Qwen processor realized to the frame-selection metadata."""

    enriched = dict(frame_metadata)
    if isinstance(reasoning_metadata, dict):
        for key in ("visual_token_count", "video_grid_thw", "padded_frame_count", "resized_height", "resized_width"):
            if reasoning_metadata.get(key) is not None:
                enriched[key] = reasoning_metadata[key]
        for key, value in reasoning_metadata.items():
            if key.startswith("qwen_") and value is not None:
                enriched[key] = value
    return enriched


# ----------------------------------------------------------------------------------------------------
# Visual budgets and Qwen token-density planning
# ----------------------------------------------------------------------------------------------------


def _adaptive_raw_video_total_pixels(config: Config) -> int:
    """Raw-pixel budget of the adaptive video streams (final answer and refiner).

    Raw frames are budgeted at ``video.raw_pixels_multiplier`` times the token budget's pixels because
    Qwen merges two frames into one temporal patch.
    """

    profile = resolve_qwen_profile(config.model.name)
    multiplier = float(config.video.raw_pixels_multiplier)
    return max(profile.min_frame_pixels, int(round(int(config.budget.total_pixels) * multiplier)))


def _full_coverage_visual_token_budget(config: Config) -> int:
    """Visual tokens a direct answer may use (the native video path needs no raw-frame multiplier)."""

    profile = resolve_qwen_profile(config.model.name)
    return max(1, int(config.budget.total_pixels) // int(profile.min_frame_pixels))


@lru_cache(maxsize=8192)
def _cached_qwen_video_dimensions(video_path: str) -> tuple[int, int] | None:
    return probe_video_dimensions(video_path)


def _qwen_direct_token_density_frame_plan(
    config: Config,
    sample: QuestionSample,
    *,
    requested_tokens_per_frame: int,
    total_frames: int,
    configured_max_frames: int,
) -> tuple[int, dict[str, Any]]:
    """Largest even frame count whose native Qwen video-path grid keeps ``requested_tokens_per_frame``.

    The processor rounds spatial sizes and pairs frames temporally, so a plain budget/density division
    can promise a density the processor does not realize. Search downward from the nominal cap and keep
    the first frame count whose predicted grid fits the token budget at the requested density.
    """

    requested_tokens_per_frame = int(requested_tokens_per_frame)
    configured_max_frames = max(0, min(int(configured_max_frames), int(total_frames)))
    profile = resolve_qwen_profile(config.model.name)
    metadata: dict[str, Any] = {
        "requested_tokens_per_frame": requested_tokens_per_frame,
        "configured_max_frames": configured_max_frames,
        "grid_aware": False,
    }
    try:
        dimensions = _cached_qwen_video_dimensions(sample.video_path)
    except Exception:
        dimensions = None
    if dimensions is None:
        metadata["fallback_reason"] = "video_dimensions_unavailable"
        return configured_max_frames, metadata

    height, width = dimensions
    temporal_size = int(profile.temporal_patch_size)
    candidate_cap = configured_max_frames // temporal_size * temporal_size
    minimum_frames = max(temporal_size, int(config.video.min_frames))
    minimum_frames = (minimum_frames + temporal_size - 1) // temporal_size * temporal_size
    video_total_pixels = int(config.budget.total_pixels)
    processor_total_pixel_budget = video_total_pixels * temporal_size
    visual_token_budget = video_total_pixels // int(profile.min_frame_pixels)
    requested_density_min_pixels = requested_tokens_per_frame * int(profile.min_frame_pixels) * temporal_size
    required_video_min_pixels = max(
        int(profile.min_frame_pixels),
        int(config.video.min_pixels),
        int(requested_density_min_pixels),
    )

    selected_plan: dict[str, Any] | None = None
    for frame_count in range(candidate_cap, minimum_frames - 1, -temporal_size):
        # qwen-vl-utils requires max_pixels >= min_pixels * 1.05; reserve that headroom inside the total
        # budget. The minimum may rise above the generic floor so low-resolution videos are upscaled to
        # the requested density.
        max_min_pixels = int(float(processor_total_pixel_budget) / (float(frame_count) * 1.05))
        if required_video_min_pixels > max_min_pixels:
            continue
        effective_min_pixels = int(required_video_min_pixels)
        minimum_processor_max_pixels = int(math.ceil(float(effective_min_pixels) * 1.05))
        processor_max_frame_pixels = min(
            int(config.video.max_pixels),
            max(int(float(processor_total_pixel_budget) / float(frame_count)), minimum_processor_max_pixels),
        )
        if processor_max_frame_pixels < minimum_processor_max_pixels:
            continue
        resized_height, resized_width = resize_size_for_profile(
            profile,
            frame_count=frame_count,
            height=int(height),
            width=int(width),
            temporal_factor=temporal_size,
            min_total_pixels=effective_min_pixels * frame_count,
            max_total_pixels=processor_max_frame_pixels * frame_count,
        )
        visual_tokens = visual_tokens_from_grid_row(
            [frame_count // temporal_size, resized_height // profile.patch_size, resized_width // profile.patch_size],
            profile,
        )
        realized_tokens_per_frame = float(visual_tokens) / float(frame_count)
        if visual_tokens > visual_token_budget:
            continue
        if realized_tokens_per_frame < requested_tokens_per_frame:
            continue
        selected_plan = {
            "target_frame_count": int(frame_count),
            "estimated_visual_token_count": int(visual_tokens),
            "estimated_visual_tokens_per_frame": realized_tokens_per_frame,
            "estimated_resized_height": int(resized_height),
            "estimated_resized_width": int(resized_width),
            "effective_video_min_pixels": int(effective_min_pixels),
            "effective_video_max_pixels": int(processor_max_frame_pixels),
        }
        break

    metadata.update(
        {
            "grid_aware": True,
            "source_height": int(height),
            "source_width": int(width),
            "processor_total_pixel_budget": int(processor_total_pixel_budget),
            "visual_token_budget": int(visual_token_budget),
            "required_video_min_pixels": int(required_video_min_pixels),
        }
    )
    if selected_plan is None:
        metadata.update({"target_frame_count": 0, "fallback_reason": "no_processor_grid_satisfies_requested_density"})
        return 0, metadata
    metadata.update(selected_plan)
    return int(selected_plan["target_frame_count"]), metadata


def _qwen_stream_token_density_frame_plan(
    config: Config,
    sample: QuestionSample,
    *,
    requested_tokens_per_frame: int,
    stream_pixel_budget: int,
    total_frames: int,
    configured_max_frames: int,
) -> tuple[int, dict[str, Any]]:
    """Largest frame count one final video stream can carry at ``requested_tokens_per_frame``.

    The stream budget is in raw pixels; one post-merge Qwen visual token covers
    ``spatial_factor**2 * temporal_patch_size`` of them. Search below the nominal budget division so grid
    rounding cannot exceed the stream's allocation.
    """

    requested_tokens_per_frame = int(requested_tokens_per_frame)
    stream_pixel_budget = max(0, int(stream_pixel_budget))
    configured_max_frames = max(0, min(int(configured_max_frames), int(total_frames)))
    profile = resolve_qwen_profile(config.model.name)
    token_cell_pixels = int(profile.min_frame_pixels) * int(profile.temporal_patch_size)
    stream_visual_token_budget = stream_pixel_budget // token_cell_pixels
    nominal_token_limited_frames = stream_visual_token_budget // max(1, requested_tokens_per_frame)
    candidate_cap = min(configured_max_frames, nominal_token_limited_frames)
    metadata: dict[str, Any] = {
        "requested_tokens_per_frame": requested_tokens_per_frame,
        "configured_max_frames": configured_max_frames,
        "stream_pixel_budget": stream_pixel_budget,
        "stream_visual_token_budget": int(stream_visual_token_budget),
        "nominal_token_limited_frame_count": int(nominal_token_limited_frames),
        "grid_aware": False,
    }
    if candidate_cap <= 0:
        metadata.update({"target_frame_count": 0, "fallback_reason": "stream_budget_below_requested_density"})
        return 0, metadata

    try:
        dimensions = _cached_qwen_video_dimensions(sample.video_path)
    except Exception:
        dimensions = None
    if dimensions is None:
        metadata.update({"target_frame_count": int(candidate_cap), "fallback_reason": "video_dimensions_unavailable"})
        return int(candidate_cap), metadata

    height, width = dimensions
    temporal_size = int(profile.temporal_patch_size)
    selected_plan: dict[str, Any] | None = None
    for frame_count in range(int(candidate_cap), 0, -1):
        padded_frame_count = (frame_count + temporal_size - 1) // temporal_size * temporal_size
        requested_min_total_pixels = min(
            stream_pixel_budget,
            frame_count * requested_tokens_per_frame * token_cell_pixels,
        )
        resized_height, resized_width = resize_size_for_profile(
            profile,
            frame_count=padded_frame_count,
            height=int(height),
            width=int(width),
            temporal_factor=temporal_size,
            min_total_pixels=requested_min_total_pixels,
            max_total_pixels=stream_pixel_budget,
        )
        visual_tokens = visual_tokens_from_grid_row(
            [padded_frame_count // temporal_size, resized_height // profile.patch_size, resized_width // profile.patch_size],
            profile,
        )
        realized_tokens_per_frame = float(visual_tokens) / float(frame_count)
        if visual_tokens > stream_visual_token_budget:
            continue
        if realized_tokens_per_frame < requested_tokens_per_frame:
            continue
        selected_plan = {
            "target_frame_count": int(frame_count),
            "estimated_visual_token_count": int(visual_tokens),
            "estimated_visual_tokens_per_frame": realized_tokens_per_frame,
            "estimated_padded_frame_count": int(padded_frame_count),
            "estimated_resized_height": int(resized_height),
            "estimated_resized_width": int(resized_width),
        }
        break

    metadata.update({"grid_aware": True, "source_height": int(height), "source_width": int(width)})
    if selected_plan is None:
        metadata.update({"target_frame_count": 0, "fallback_reason": "no_stream_processor_grid_satisfies_requested_density"})
        return 0, metadata
    metadata.update(selected_plan)
    return int(selected_plan["target_frame_count"]), metadata


def _policy_name(allocation: AdaptiveAllocation) -> str:
    return str(allocation.metadata.get("planner_policy_name") or "").strip().upper()


def _qwen_overview_density_controls_for_allocation(
    config: Config,
    allocation: AdaptiveAllocation,
) -> tuple[int | None, int]:
    """Final-overview tokens-per-frame floor (None: no floor) and frame cap for the selected policy."""

    policy_name = _policy_name(allocation)
    if policy_name == "LOCAL_ONLY":
        # LOCAL_ONLY gives the overview pixels only when it has no local evidence, so the whole budget
        # goes to the overview; it then samples the overview exactly as GLOBAL_ONLY does.
        policy_name = "GLOBAL_ONLY"
    return config.tokens_per_frame.overview.get(policy_name), int(config.caps.overview[policy_name])


def _qwen_local_density_target_for_allocation(config: Config, allocation: AdaptiveAllocation) -> int | None:
    """Final-local tokens-per-frame floor for the selected policy (None: no floor)."""

    target = config.tokens_per_frame.local.get(_policy_name(allocation))
    return int(target) if target is not None else None


# ----------------------------------------------------------------------------------------------------
# Local (detail) frame sampling
# ----------------------------------------------------------------------------------------------------


def _local_sampling_configured_max_frames(config: Config) -> int:
    return max(0, min(int(config.caps.refiner), int(config.caps.local)))


def _local_sampling_min_feasible_frame_cap(local_pixels: int | None) -> int | None:
    if local_pixels is None:
        return None
    return max(0, int(local_pixels) // MIN_FEASIBLE_LOCAL_FRAME_PIXELS)


def _local_sampling_max_frames(config: Config, *, local_pixels: int | None = None) -> int:
    max_frames = _local_sampling_configured_max_frames(config)
    pixel_limited_frames = _local_sampling_min_feasible_frame_cap(local_pixels)
    if pixel_limited_frames is not None:
        max_frames = min(max_frames, pixel_limited_frames)
    return max(0, max_frames)


def _select_region_only_reasoning_indices(
    config: Config,
    retrieved_segments: list[RetrievedSegment],
    *,
    fps: float,
    total_frames: int,
    local_pixels: int | None = None,
    max_frames_override: int | None = None,
) -> list[int]:
    """Local-stream frames: ``fps.local`` inside the segments, frames split across segments by score."""

    max_frames = _local_sampling_max_frames(config, local_pixels=local_pixels)
    if max_frames_override is not None:
        max_frames = min(max_frames, max(0, int(max_frames_override)))
    return sample_frames_from_retrieved_segments(
        retrieved_segments,
        native_fps=fps,
        sample_fps=config.fps.local or fps,
        total_frames=total_frames,
        max_frames=max_frames,
    )


def _region_only_reasoning_sampling_metadata(
    config: Config,
    *,
    fps: float,
    local_pixels: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "local_sampling_fps": float(config.fps.local or fps),
        "local_sampling_configured_max_frames": int(_local_sampling_configured_max_frames(config)),
        "local_sampling_max_frames": int(_local_sampling_max_frames(config, local_pixels=local_pixels)),
    }
    if local_pixels is not None:
        metadata.update(
            {
                "local_sampling_pixel_budget": int(local_pixels),
                "local_sampling_min_feasible_frame_cap": int(_local_sampling_min_feasible_frame_cap(local_pixels) or 0),
                "local_sampling_policy": "sample_by_fps_then_resize_to_fit_budget",
            }
        )
    return metadata


# ----------------------------------------------------------------------------------------------------
# Video-stream evidence
# ----------------------------------------------------------------------------------------------------


def _stream_video_metadata(
    *,
    sample: QuestionSample,
    frame_refs: list[FrameReference],
    fps: float,
    total_frames: int,
    kind: str,
    label: str,
    total_pixels: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame_indices = [int(frame.frame_index) for frame in frame_refs]
    timestamps = [float(frame.timestamp_sec) for frame in frame_refs]
    duration = sample.video_metadata.duration_sec
    metadata: dict[str, Any] = {
        "fps": float(fps),
        "frames_indices": frame_indices,
        "timestamps_sec": timestamps,
        "total_num_frames": int(total_frames),
        "duration": float(duration) if duration is not None else (float(total_frames) / float(fps) if fps else None),
        "height": None,
        "width": None,
        "video_backend": VIDEO_STREAM_BACKEND,
        "stream_kind": kind,
        "stream_label": label,
        "visual_budget_total_pixels": int(total_pixels),
        "source_video_id": sample.video_id,
        "source_question_uid": sample.question_uid,
        "timestamp_range_sec": [min(timestamps), max(timestamps)] if timestamps else None,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _split_segment_stream_groups(
    refs: list[FrameReference],
    frames: list[Any],
    stream_context: dict[str, Any] | None,
) -> list[tuple[int, list[FrameReference], list[Any], dict[str, Any]]]:
    """Group frames into one clip per segment; earlier segments claim shared frames first."""

    if not refs or not frames:
        return []
    context = stream_context or {}
    ranges = context.get("local_segment_time_ranges_sec") or []
    ids = context.get("local_segment_ids") or []
    sources = context.get("local_segment_sources") or []
    queries = context.get("local_segment_queries") or []
    scores = context.get("local_segment_scores") or []
    frame_indices_by_segment = context.get("local_segment_frame_indices") or []
    if not ranges:
        return [(1, list(refs), list(frames), dict(context))]

    assigned: set[int] = set()
    groups: list[tuple[int, list[FrameReference], list[Any], dict[str, Any]]] = []
    pairs = list(zip(refs, frames))
    for segment_index, time_range in enumerate(ranges, start=1):
        try:
            start_sec, end_sec = float(time_range[0]), float(time_range[1])
        except (TypeError, ValueError, IndexError):
            continue
        explicit_indices = set()
        if segment_index - 1 < len(frame_indices_by_segment):
            explicit_indices = {int(index) for index in frame_indices_by_segment[segment_index - 1]}
        segment_pairs: list[tuple[FrameReference, Any]] = []
        for pair_index, (frame_ref, frame) in enumerate(pairs):
            if pair_index in assigned:
                continue
            in_time_range = start_sec <= float(frame_ref.timestamp_sec) <= end_sec
            in_explicit_indices = bool(explicit_indices) and int(frame_ref.frame_index) in explicit_indices
            if in_time_range or in_explicit_indices:
                assigned.add(pair_index)
                segment_pairs.append((frame_ref, frame))
        if not segment_pairs:
            continue
        metadata: dict[str, Any] = {
            "segment_id": ids[segment_index - 1] if segment_index - 1 < len(ids) else f"segment_{segment_index}",
            "segment_time_range_sec": [start_sec, end_sec],
        }
        if segment_index - 1 < len(sources):
            metadata["segment_source"] = sources[segment_index - 1]
        if segment_index - 1 < len(queries):
            metadata["segment_query"] = queries[segment_index - 1]
        if segment_index - 1 < len(scores):
            metadata["segment_score"] = float(scores[segment_index - 1])
        if explicit_indices:
            metadata["segment_frame_indices"] = sorted(explicit_indices)
        groups.append((segment_index, [ref for ref, _ in segment_pairs], [frame for _, frame in segment_pairs], metadata))

    remaining_pairs = [(ref, frame) for pair_index, (ref, frame) in enumerate(pairs) if pair_index not in assigned]
    if remaining_pairs:
        metadata = {"segment_id": "segment_unmatched", "segment_time_range_sec": None}
        groups.append((len(groups) + 1, [ref for ref, _ in remaining_pairs], [frame for _, frame in remaining_pairs], metadata))
    return groups or [(1, list(refs), list(frames), dict(context))]


def _allocate_segment_stream_pixels(total_pixels: int, frame_counts: list[int]) -> list[int]:
    if not frame_counts:
        return []
    total_pixels = max(0, int(total_pixels))
    total_frames = max(1, sum(max(0, count) for count in frame_counts))
    allocations: list[int] = []
    remaining = total_pixels
    for index, frame_count in enumerate(frame_counts):
        if index == len(frame_counts) - 1:
            allocation = max(0, remaining)
        else:
            allocation = int(total_pixels * max(0, frame_count) / total_frames)
            remaining -= allocation
        allocations.append(allocation)
    return allocations


def _split_temporally_discontinuous_stream_groups(
    groups: list[tuple[int, list[FrameReference], list[Any], dict[str, Any]]],
    *,
    expected_sampling_fps: float | None,
) -> list[tuple[int, list[FrameReference], list[Any], dict[str, Any]]]:
    """Split sampled clips at gaps that are large relative to their normal cadence."""

    split_groups: list[tuple[int, list[FrameReference], list[Any], dict[str, Any]]] = []
    for segment_index, refs, frames, metadata in groups:
        pairs = list(zip(refs, frames))
        if len(pairs) < 2:
            split_groups.append((segment_index, list(refs), list(frames), dict(metadata)))
            continue

        positive_deltas = sorted(
            current.timestamp_sec - previous.timestamp_sec
            for previous, current in zip(refs, refs[1:])
            if current.timestamp_sec > previous.timestamp_sec
        )
        if not positive_deltas:
            split_groups.append((segment_index, list(refs), list(frames), dict(metadata)))
            continue

        lower_quartile_index = (len(positive_deltas) - 1) // 4
        typical_delta = float(positive_deltas[lower_quartile_index])
        if expected_sampling_fps is not None and expected_sampling_fps > 0:
            typical_delta = max(typical_delta, 1.0 / float(expected_sampling_fps))
        discontinuity_ratio = 2.0
        initial_discontinuity_threshold_sec = typical_delta * discontinuity_ratio

        runs: list[list[tuple[FrameReference, Any]]] = [[]]
        current_cadence_sec = typical_delta
        for pair_position, pair in enumerate(pairs):
            if pair_position > 0:
                gap_sec = pair[0].timestamp_sec - pairs[pair_position - 1][0].timestamp_sec
                if gap_sec > current_cadence_sec * discontinuity_ratio + 1e-6:
                    runs.append([])
                    # A new run may have its own cadence: seed it from the boundary gap, then let a smaller
                    # following gap establish the actual cadence.
                    current_cadence_sec = gap_sec
                elif gap_sec * discontinuity_ratio + 1e-6 < current_cadence_sec:
                    current_cadence_sec = gap_sec
            runs[-1].append(pair)

        for run_position, run in enumerate(runs, start=1):
            run_refs = [frame_ref for frame_ref, _ in run]
            run_metadata = {
                **metadata,
                "parent_segment_stream_index": int(segment_index),
                "temporal_subclip_index": run_position,
                "temporal_subclip_count": len(runs),
                "temporal_discontinuity_initial_threshold_sec": initial_discontinuity_threshold_sec,
                "temporal_discontinuity_ratio": discontinuity_ratio,
                "segment_time_range_sec": [float(run_refs[0].timestamp_sec), float(run_refs[-1].timestamp_sec)],
            }
            split_groups.append((segment_index, run_refs, [frame for _, frame in run], run_metadata))
    return split_groups


def _build_hybrid_video_stream_input(
    *,
    config: Config,
    sample: QuestionSample,
    total_frames: int,
    fps: float,
    overview_refs: list[FrameReference],
    overview_frames: list[Any],
    segment_refs: list[FrameReference],
    segment_frames: list[Any],
    overview_pixels: int,
    segment_pixels: int,
    stream_context: dict[str, Any] | None = None,
    qwen_overview_min_visual_tokens_per_frame: int | None = None,
    qwen_local_min_visual_tokens_per_frame: int | None = None,
) -> VisualEvidenceInput:
    """One overview stream plus one detail stream per segment clip, local pixels split by frame count."""

    streams: list[VideoEvidenceStream] = []
    if overview_frames:
        overview_extra: dict[str, Any] = {"stream_description": "uniform low-resolution context from the full source video"}
        if qwen_overview_min_visual_tokens_per_frame is not None:
            overview_extra.update(
                {
                    "qwen_min_visual_tokens_per_frame": int(qwen_overview_min_visual_tokens_per_frame),
                    "qwen_overview_min_visual_tokens_per_frame": int(qwen_overview_min_visual_tokens_per_frame),
                }
            )
        streams.append(
            VideoEvidenceStream(
                label="Overview video",
                kind="overview",
                frames=list(overview_frames),
                frame_references=list(overview_refs),
                total_pixels=int(overview_pixels),
                metadata=_stream_video_metadata(
                    sample=sample,
                    frame_refs=overview_refs,
                    fps=fps,
                    total_frames=total_frames,
                    kind="overview",
                    label="Overview video",
                    total_pixels=int(overview_pixels),
                    extra=overview_extra,
                ),
            )
        )
    segment_groups = _split_segment_stream_groups(segment_refs, segment_frames, stream_context)
    segment_groups = _split_temporally_discontinuous_stream_groups(segment_groups, expected_sampling_fps=config.fps.local or fps)
    segment_allocations = _allocate_segment_stream_pixels(
        int(segment_pixels),
        [len(group_frames) for _, _, group_frames, _ in segment_groups],
    )
    label_position = 0
    for group_index, (segment_index, group_refs, group_frames, group_metadata) in enumerate(segment_groups):
        if not group_frames:
            continue
        label_position += 1
        group_pixels = segment_allocations[group_index]
        label = f"Detail video {label_position}"
        extra: dict[str, Any] = {
            "stream_description": "high-detail temporal evidence from one retrieved/local segment",
            "segment_stream_index": int(segment_index),
            "segment_stream_count": len(segment_groups),
        }
        if qwen_local_min_visual_tokens_per_frame is not None:
            extra.update(
                {
                    "qwen_min_visual_tokens_per_frame": int(qwen_local_min_visual_tokens_per_frame),
                    "qwen_local_min_visual_tokens_per_frame": int(qwen_local_min_visual_tokens_per_frame),
                }
            )
        extra.update(group_metadata)
        streams.append(
            VideoEvidenceStream(
                label=label,
                kind="segment",
                frames=list(group_frames),
                frame_references=list(group_refs),
                total_pixels=int(group_pixels),
                metadata=_stream_video_metadata(
                    sample=sample,
                    frame_refs=group_refs,
                    fps=fps,
                    total_frames=total_frames,
                    kind="segment",
                    label=label,
                    total_pixels=int(group_pixels),
                    extra=extra,
                ),
            )
        )

    return VisualEvidenceInput(
        frames=[frame for stream in streams for frame in stream.frames],
        mode="video_streams",
        groups=[stream.kind for stream in streams for _ in stream.frames],
        group_total_pixels={"overview": int(overview_pixels), "segment": int(segment_pixels)},
        video_streams=streams,
    )


def _build_refinement_video_stream_input(
    config: Config,
    *,
    sample: QuestionSample,
    memory_regions: list[Any],
    candidate_refs: list[FrameReference],
    candidate_frames: list[Any],
    total_frames: int,
    fps: float,
    region_frame_indices: dict[str, list[int]] | None = None,
) -> VisualEvidenceInput:
    """Encode each refinement region as a coherent native video stream.

    With ``region_frame_indices``, each region's clip takes exactly those frames and any leftover frames
    are labelled as context rather than as a region. Annotates ``candidate_refs`` in place.
    """

    explicit_context = (
        {
            "local_segment_frame_indices": [
                list(region_frame_indices.get(str(region.region_id), [])) for region in memory_regions
            ]
        }
        if region_frame_indices is not None
        else {}
    )
    raw_input = _build_hybrid_video_stream_input(
        config=config,
        sample=sample,
        total_frames=total_frames,
        fps=fps,
        overview_refs=[],
        overview_frames=[],
        segment_refs=candidate_refs,
        segment_frames=candidate_frames,
        overview_pixels=0,
        segment_pixels=_adaptive_raw_video_total_pixels(config),
        stream_context={
            "local_segment_ids": [str(region.region_id) for region in memory_regions],
            "local_segment_sources": ["+".join(str(source) for source in region.sources) for region in memory_regions],
            "local_segment_queries": [
                str(region.queries[0]) if region.queries else str(region.rationale or region.region_id)
                for region in memory_regions
            ],
            "local_segment_scores": [float(max(region.score, region.confidence)) for region in memory_regions],
            "local_segment_time_ranges_sec": [[float(region.start_sec), float(region.end_sec)] for region in memory_regions],
            **explicit_context,
        },
    )
    for stream in raw_input.video_streams:
        region_id = str(stream.metadata.get("segment_id") or stream.label)
        if region_frame_indices is not None and region_id == "segment_unmatched":
            stream.label = "Context frames outside every memory region"
            stream.metadata.update(
                {
                    "stream_label": stream.label,
                    "stream_description": "context frames; not a memory-bank region",
                    "memory_region_id": None,
                }
            )
            for frame_ref in stream.frame_references:
                frame_ref.evidence_type = "refinement_candidate"
                frame_ref.evidence_role = "refinement_context"
                frame_ref.evidence_region = None
            continue
        stream.label = f"Candidate region {region_id}"
        stream.metadata.update(
            {
                "stream_label": stream.label,
                "stream_description": "coherent candidate clip for memory-bank refinement",
                "memory_region_id": region_id,
            }
        )
        for frame_ref in stream.frame_references:
            frame_ref.evidence_type = "refinement_candidate"
            frame_ref.evidence_role = "candidate_region_visual_check"
            frame_ref.evidence_region = region_id
    return raw_input


# ----------------------------------------------------------------------------------------------------
# Uniform-frame answers
# ----------------------------------------------------------------------------------------------------


def _adaptive_full_coverage_direct_indices(
    config: Config,
    *,
    total_frames: int,
    fps: float,
    sample: QuestionSample,
    force_direct: bool = False,
    decision_metadata: dict[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]] | None:
    if not config.adaptive_allocation and not force_direct:
        return None

    sample_fps = float(config.full_coverage.fps)
    configured_max_frames_before_density = min(int(config.caps.full_coverage), int(total_frames))
    full_coverage_visual_token_budget = _full_coverage_visual_token_budget(config)
    stride_threshold_sec = float(config.full_coverage.max_stride_sec)
    duration_sec = float(total_frames) / float(fps)

    density_evaluations: list[dict[str, Any]] = []
    evaluated_plans: list[tuple[int, int, dict[str, Any], bool]] = []
    for density_candidate in config.full_coverage.tokens_per_frame_ladder:
        candidate_nominal_frame_cap = int(full_coverage_visual_token_budget) // int(density_candidate)
        candidate_max_frames, candidate_plan = _qwen_direct_token_density_frame_plan(
            config,
            sample,
            requested_tokens_per_frame=int(density_candidate),
            total_frames=total_frames,
            configured_max_frames=min(configured_max_frames_before_density, candidate_nominal_frame_cap),
        )
        candidate_global_frame_count = 0
        candidate_global_stride_sec: float | None = None
        candidate_within_stride_threshold = False
        if candidate_max_frames > 0:
            # The stride is measured on the candidate's own frame cap.
            candidate_reference_frames = min(int(candidate_max_frames), int(total_frames))
            candidate_global_indices = qwen_fps_frame_indices(
                total_frames,
                fps,
                sample_fps,
                candidate_reference_frames,
                min_frames=min(int(config.video.min_frames), int(candidate_max_frames)),
            )
            candidate_global_frame_count = len(candidate_global_indices)
            candidate_global_stride_sec = duration_sec / max(candidate_global_frame_count - 1, 1)
            candidate_within_stride_threshold = candidate_global_stride_sec <= stride_threshold_sec
        density_evaluations.append(
            {
                "requested_visual_tokens_per_frame": int(density_candidate),
                "nominal_frame_cap": int(candidate_nominal_frame_cap),
                "processor_grid_frame_cap": int(candidate_max_frames),
                "processor_estimated_visual_tokens_per_frame": candidate_plan.get("estimated_visual_tokens_per_frame"),
                "global_frame_count": int(candidate_global_frame_count),
                "global_stride_sec": candidate_global_stride_sec,
                "within_stride_threshold": bool(candidate_within_stride_threshold),
            }
        )
        evaluated_plans.append((int(density_candidate), int(candidate_max_frames), candidate_plan, bool(candidate_within_stride_threshold)))

    usable_plans = [candidate for candidate in evaluated_plans if candidate[1] > 0]
    if force_direct and usable_plans:
        selected_density_plan = usable_plans[-1]
        density_selection_reason = "forced_direct_lowest_density_for_coverage"
    else:
        selected_density_plan = next((candidate for candidate in usable_plans if candidate[3]), None)
        if selected_density_plan is not None:
            density_selection_reason = "densest_candidate_within_stride_threshold"
        elif usable_plans:
            selected_density_plan = usable_plans[-1]
            density_selection_reason = "lowest_density_candidate_exceeds_stride_threshold"
        else:
            selected_density_plan = evaluated_plans[-1]
            density_selection_reason = "no_candidate_has_valid_processor_grid"
    selected_min_visual_tokens_per_frame, max_frames, qwen_direct_frame_plan, _ = selected_density_plan
    token_density_frame_cap = int(full_coverage_visual_token_budget) // int(selected_min_visual_tokens_per_frame)
    configured_max_frames = min(configured_max_frames_before_density, token_density_frame_cap)

    forced_direct_density_fallback = False
    if (
        max_frames <= 0
        and force_direct
        and configured_max_frames > 0
        and qwen_direct_frame_plan.get("fallback_reason") == "no_processor_grid_satisfies_requested_density"
    ):
        # A low-resolution video may not reach the requested density without upscaling. A forced
        # fallback must still answer; the nominal cap is budget-safe because the processor grid is then
        # less dense than requested.
        max_frames = int(configured_max_frames)
        forced_direct_density_fallback = True
        qwen_direct_frame_plan.update(
            {
                "forced_direct_density_fallback": True,
                "density_fallback_frame_count": int(max_frames),
                "density_fallback_reason": "native_processor_grid_below_requested_density",
            }
        )

    metadata: dict[str, Any] = {
        "enabled": bool(config.adaptive_allocation),
        "triggered": False,
        "sample_fps": sample_fps,
        "max_frames": int(max_frames),
        "total_frames": int(total_frames),
        "native_fps": float(fps),
        "reason": None,
        "configured_max_frames_before_token_density": int(configured_max_frames_before_density),
        "full_coverage_visual_token_budget": int(full_coverage_visual_token_budget),
        "full_coverage_configured_min_visual_tokens_per_frame": int(config.full_coverage.min_tokens_per_frame),
        "full_coverage_visual_tokens_per_frame_candidates": list(config.full_coverage.tokens_per_frame_ladder),
        "full_coverage_density_candidate_evaluations": density_evaluations,
        "full_coverage_density_selection_reason": density_selection_reason,
        "full_coverage_min_visual_tokens_per_frame": int(selected_min_visual_tokens_per_frame),
        "full_coverage_token_density_frame_cap": int(token_density_frame_cap),
        "full_coverage_effective_max_frames": int(max_frames),
        "qwen_direct_frame_plan": qwen_direct_frame_plan,
    }
    if forced_direct_density_fallback:
        metadata["full_coverage_forced_direct_density_fallback"] = True

    def _finish(selected_indices: list[int] | None) -> tuple[list[int], dict[str, Any]] | None:
        if selected_indices:
            selected_nominal_density = float(full_coverage_visual_token_budget) / float(len(selected_indices))
            metadata.update(
                {
                    "full_coverage_selected_nominal_visual_tokens_per_frame": selected_nominal_density,
                    "full_coverage_min_visual_tokens_per_frame_satisfied": (
                        selected_nominal_density >= float(selected_min_visual_tokens_per_frame)
                    ),
                    "full_coverage_density_satisfaction_basis": "nominal_budget_division",
                }
            )
        if decision_metadata is not None:
            decision_metadata.clear()
            decision_metadata.update(metadata)
        if selected_indices is None:
            return None
        return selected_indices, metadata

    if max_frames <= 0:
        metadata["reason"] = "full_coverage_token_density_exceeds_visual_budget"
        return _finish(None)

    min_frames = min(int(config.video.min_frames), max_frames)

    def _select_direct_indices() -> tuple[list[int], dict[str, Any]]:
        target_frame_count = min(int(max_frames), int(total_frames), max(1, int(duration_sec * sample_fps)))
        selected_indices = qwen_fps_frame_indices(
            total_frames,
            fps,
            sample_fps,
            target_frame_count,
            min_frames=min(min_frames, target_frame_count),
        )
        return selected_indices, {
            "sampling_mode": "uniform_up_to_max_frames",
            "target_frame_count": int(target_frame_count),
            "target_sample_fps": sample_fps,
            "selected_frame_count": len(selected_indices),
        }

    if force_direct:
        selected_indices, selection_metadata = _select_direct_indices()
        metadata.update(selection_metadata)
        metadata.update({"forced_direct_answer": True, "triggered": True, "reason": "forced_direct_answer"})
        return _finish(selected_indices)

    reference_frames = min(int(max_frames), int(total_frames))
    global_indices = qwen_fps_frame_indices(total_frames, fps, sample_fps, reference_frames, min_frames=min_frames)
    global_stride_sec = duration_sec / max(len(global_indices) - 1, 1)
    metadata.update(
        {
            "full_coverage_global_stride_sec": float(global_stride_sec),
            "full_coverage_global_stride_threshold_sec": stride_threshold_sec,
            "full_coverage_global_frame_count": len(global_indices),
            "full_coverage_reference_frame_count": int(reference_frames),
        }
    )
    if global_stride_sec <= stride_threshold_sec:
        selected_indices, selection_metadata = _select_direct_indices()
        metadata.update(selection_metadata)
        metadata.update({"triggered": True, "reason": "full_coverage_global_stride_within_threshold"})
        return _finish(selected_indices)

    metadata["reason"] = "full_coverage_global_stride_exceeds_threshold"
    return _finish(None)


def _augment_full_coverage_processor_density(
    full_coverage_metadata: dict[str, Any],
    reasoning_metadata: dict[str, Any] | None,
) -> None:
    """Attach the density the native Qwen processor actually realized."""

    if full_coverage_metadata.get("triggered") is not True or not isinstance(reasoning_metadata, dict):
        return
    visual_tokens = reasoning_metadata.get("visual_token_count")
    processor_frames = reasoning_metadata.get("qwen_video_path_actual_frame_count")
    if not isinstance(processor_frames, (int, float)) or processor_frames <= 0:
        processor_frames = reasoning_metadata.get("padded_frame_count")
    if (
        not isinstance(visual_tokens, (int, float))
        or visual_tokens <= 0
        or not isinstance(processor_frames, (int, float))
        or processor_frames <= 0
    ):
        return
    realized_density = float(visual_tokens) / float(processor_frames)
    full_coverage_metadata.update(
        {
            "full_coverage_processor_visual_token_count": int(visual_tokens),
            "full_coverage_processor_frame_count": int(processor_frames),
            "full_coverage_realized_visual_tokens_per_frame": realized_density,
            "full_coverage_density_satisfaction_basis": "processor_realized_grid",
        }
    )
    requested_density = full_coverage_metadata.get("full_coverage_min_visual_tokens_per_frame")
    if isinstance(requested_density, (int, float)):
        full_coverage_metadata["full_coverage_min_visual_tokens_per_frame_satisfied"] = realized_density >= float(requested_density)


def _answer_question_with_prompt_style(
    reasoner: Any,
    config: Config,
    sample: QuestionSample,
    selected_frames: list[FrameReference],
    raw_frames: Any,
    retrieved_segments: list[RetrievedSegment],
    *,
    use_video_path_preprocessing: bool = False,
    qwen_video_path_frame_plan: dict[str, Any] | None = None,
) -> AnswerPrediction:
    """Answer with the ``simple_mcq`` prompt, optionally through Qwen's native video reader.

    The overrides are thread-local to this call (``reasoner.answer_overrides``).
    """

    overrides: dict[str, Any] = {"prompt_style": SIMPLE_MCQ_PROMPT_STYLE}
    if use_video_path_preprocessing:
        overrides["video_path_fps"] = config.full_coverage.fps
        if selected_frames:
            planned_min_pixels = (
                qwen_video_path_frame_plan.get("effective_video_min_pixels")
                if isinstance(qwen_video_path_frame_plan, dict)
                else None
            )
            if planned_min_pixels is not None:
                overrides["video_min_pixels"] = int(planned_min_pixels)
            else:
                # No grid plan: lower the floor only as far as qwen-vl-utils' max >= 1.05 * min needs.
                profile = resolve_qwen_profile(config.model.name)
                max_min_pixels = int(
                    (float(config.budget.total_pixels) * float(profile.temporal_patch_size))
                    / (float(len(selected_frames)) * 1.05)
                )
                overrides["video_min_pixels"] = min(
                    int(config.video.min_pixels),
                    max(profile.min_frame_pixels, max_min_pixels),
                )
        overrides["video_max_frames"] = max(int(config.video.min_frames), int(len(selected_frames)))
    with reasoner.answer_overrides(**overrides):
        return reasoner.answer_question(sample, selected_frames, raw_frames, retrieved_segments)


def _adaptive_plan_parse_failure_uniform_indices(
    config: Config,
    *,
    total_frames: int,
) -> tuple[list[int], dict[str, Any]]:
    max_frames = max(1, min(int(config.caps.planner), int(total_frames)))
    selected_indices = uniform_frame_indices(total_frames, max_frames)
    return selected_indices, {
        "enabled": True,
        "triggered": True,
        "reason": "adaptive_overview_plan_parse_failed",
        "sampling_mode": "uniform",
        "max_frames": int(max_frames),
        "selected_frame_count": len(selected_indices),
        "total_frames": int(total_frames),
        "answer_prompt_style": SIMPLE_MCQ_PROMPT_STYLE,
    }


# ----------------------------------------------------------------------------------------------------
# Planner and retrieval
# ----------------------------------------------------------------------------------------------------


def _prepare_adaptive_overview_inputs(
    config: Config,
    sample: QuestionSample,
    *,
    total_frames: int,
    fps: float,
) -> tuple[list[FrameReference], list, dict[str, Any]]:
    """The planner's sparse overview: ``fps.overview`` across the video, at most ``frames.planner`` frames."""

    overview_max_frames = min(int(config.caps.planner), total_frames)
    overview_indices = qwen_fps_frame_indices(total_frames, fps, config.fps.overview, overview_max_frames, min_frames=1)
    refs, frames, metadata = _prepare_reasoning_inputs(sample, overview_indices, fps=fps)
    metadata["adaptive_overview_source"] = "overview_fps_sampling"
    return refs, frames, metadata


def _apply_question_time_interval_override(sample: QuestionSample, plan: AdaptiveOverviewPlan) -> AdaptiveOverviewPlan:
    """A question that names a time interval is answered LOCAL_ONLY on that interval, without retrieval."""

    spans, _ = parse_time_reference(sample.question_text)
    plan.temporal_proposals = [proposal for proposal in plan.temporal_proposals if proposal.source != "question"]
    if not spans:
        return plan

    for start_sec, end_sec in spans:
        plan.temporal_proposals.append(
            AdaptiveTemporalProposal(
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                label="specified question interval",
                confidence=1.0,
                rationale="The question text explicitly specifies this time interval.",
                source="question",
            )
        )
    plan.local_retrieval_needed = False
    plan.local_retrieval_reason = (
        "The question explicitly specifies the time interval to inspect, so UniversalVTG retrieval is not needed."
    )
    plan.queries = []
    plan.evidence_needs = list(dict.fromkeys([*plan.evidence_needs, "inspect the explicitly specified question interval"]))
    plan.budget_policy.local_evidence_needed = True
    plan.budget_policy.policy_name = "LOCAL_ONLY"
    if plan.budget_policy.local_detail_level in {None, "NONE"}:
        plan.budget_policy.local_detail_level = "STANDARD"
    plan.budget_policy.needs_global_context_for_refinement = False
    plan.budget_policy.overview_min_ratio = 0.0
    plan.budget_policy.local_budget_ratio = 1.0
    plan.budget_policy.overview_ratio = 0.0
    plan.budget_policy.local_ratio = 1.0
    return plan


def _normalized_query_key(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _normalized_text_tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip().split()


def _query_copies_option(query: str, options: list[str]) -> bool:
    query_text = " " + " ".join(_normalized_text_tokens(query)) + " "
    for option in options:
        option_tokens = _normalized_text_tokens(option)
        if len(option_tokens) < 4:
            continue
        window = min(5, len(option_tokens))
        for index in range(0, len(option_tokens) - window + 1):
            phrase = " ".join(option_tokens[index : index + window])
            if len(phrase) >= 12 and f" {phrase} " in query_text:
                return True
    return False


def _query_is_negative_or_absence_target(query: str) -> bool:
    query_text = " " + " ".join(_normalized_text_tokens(query)) + " "
    negative_patterns = [
        r"\bnot (?:visible|shown|seen|discussed|mentioned|present|found)\b",
        r"\bwas not (?:visible|shown|seen|discussed|mentioned|present|found)\b",
        r"\bwere not (?:visible|shown|seen|discussed|mentioned|present|found)\b",
        r"\bno [a-z0-9 ]{1,48} (?:was |were |is |are )?(?:visible|shown|seen|found|present|discussed|mentioned)\b",
    ]
    return any(re.search(pattern, query_text) for pattern in negative_patterns)


def _sanitize_generated_retrieval_queries(
    sample: QuestionSample,
    generated_queries: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Drop duplicate queries, queries that copy or restate an option, and queries for absent things."""

    used_queries: list[str] = []
    seen: set[str] = set()
    rejected: list[dict[str, str]] = []
    for query in generated_queries:
        query_key = _normalized_query_key(query)
        if not query_key:
            continue
        reason = None
        if query_key in seen:
            reason = "duplicate"
        elif _query_copies_option(query, sample.options):
            reason = "option_copy"
        elif _query_is_negative_or_absence_target(query):
            reason = "negative_or_absence_target"
        elif option_restated_by_query(query, sample.options, sample.question_text) is not None:
            reason = "option_restatement"
        if reason is not None:
            rejected.append({"query": query, "reason": reason})
            continue
        seen.add(query_key)
        used_queries.append(query)

    return used_queries, {
        "input_queries": list(generated_queries),
        "used_queries": list(used_queries),
        "rejected_queries": rejected,
        "fallback_to_original_question": bool(generated_queries and not used_queries),
    }


def _adaptive_retrieval_query_sets(
    sample: QuestionSample,
    plan: AdaptiveOverviewPlan,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Sanitized planner queries, the question itself (unless a planner query repeats it), and both."""

    generated_queries = [query.strip() for query in plan.queries if query and query.strip()]
    generated_queries, sanitizer_metadata = _sanitize_generated_retrieval_queries(sample, generated_queries)
    seen = {_normalized_query_key(query) for query in generated_queries}
    question_queries: list[str] = []
    question_query = sample.question_text.strip()
    if question_query and _normalized_query_key(question_query) not in seen:
        question_queries.append(question_query)
    return generated_queries, question_queries, [*generated_queries, *question_queries], sanitizer_metadata


def _mark_question_retrieval_segments(
    segments: list[RetrievedSegment],
    question_queries: list[str],
) -> list[RetrievedSegment]:
    question_query_keys = {_normalized_query_key(query) for query in question_queries}
    return [
        RetrievedSegment(
            query=segment.query,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            score=segment.score,
            frame_indices=list(segment.frame_indices),
            source=(
                "universal_vtg_original_question"
                if _normalized_query_key(segment.query) in question_query_keys
                else segment.source
            ),
        )
        for segment in segments
    ]


def _apply_adaptive_retrieval_confidence_gate(
    config: Config,
    segments: list[RetrievedSegment],
) -> tuple[list[RetrievedSegment], bool, dict[str, Any], list[str]]:
    """Gate retrieval by confidence.

    When every retrieved moment scores below ``memory.weak_retrieval_threshold`` the question is answered
    from uniform frames (returns no segments and True). Otherwise moments below
    ``memory.min_region_confidence`` are dropped. Also returns metadata and fallback reasons.
    """

    original_segments = list(segments)
    original_count = len(original_segments)
    scores = [float(segment.score) for segment in original_segments]
    max_score = max(scores) if scores else None
    all_weak_threshold = float(config.memory.weak_retrieval_threshold)
    min_region_confidence = float(config.memory.min_region_confidence)
    metadata: dict[str, Any] = {
        "all_weak_threshold": all_weak_threshold,
        "min_region_confidence": min_region_confidence,
        "retrieved_segment_count_before_filtering": original_count,
        "max_retrieval_score": max_score,
        "triggered": False,
        "trigger_reason": None,
    }
    if original_count > 0 and max_score is not None and max_score < all_weak_threshold:
        metadata.update(
            {
                "triggered": True,
                "trigger_reason": "all_retrieval_scores_below_threshold",
                "retrieved_segment_count_after_filtering": 0,
            }
        )
        return [], True, metadata, ["all_weak_retrieval_direct_answer"]

    fallback_reasons: list[str] = []
    filtered_segments = original_segments
    if min_region_confidence > 0:
        filtered_segments = [segment for segment in original_segments if float(segment.score) >= min_region_confidence]
        if len(filtered_segments) != original_count:
            fallback_reasons.append("weak_retrieval_segments_filtered")
    metadata["retrieved_segment_count_after_filtering"] = len(filtered_segments)
    return filtered_segments, False, metadata, fallback_reasons


# ----------------------------------------------------------------------------------------------------
# Final answer input
# ----------------------------------------------------------------------------------------------------


def _prepare_adaptive_final_reasoning_inputs(
    config: Config,
    sample: QuestionSample,
    local_segments: list[RetrievedSegment],
    allocation: AdaptiveAllocation,
    *,
    total_frames: int,
    fps: float,
) -> tuple[list[FrameReference], VisualEvidenceInput, dict[str, Any]]:
    """Overview and detail streams for the final answer at the allocation's pixel split.

    The overview samples ``fps.overview`` across the video up to the policy's frame cap and, for a policy
    with a tokens-per-frame floor, only as many frames as keep that floor within its pixels. The local
    stream samples ``fps.local`` inside the memory regions up to ``frames.local`` (LOCAL_HEAVY also keeps a
    floor) and is resized to fit its pixels; overview frames next to a local frame are dropped.
    Updates ``allocation`` with the final split.
    """

    overview_sampling_fps = config.fps.overview
    overview_min_visual_tokens_per_frame: int | None = None
    overview_configured_max_frames = 0
    overview_frame_cap = 0
    overview_density_plan: dict[str, Any] | None = None
    # LOCAL_ONLY with local evidence has no overview pixels and no overview stream.
    if int(allocation.overview_pixels) > 0:
        overview_min_visual_tokens_per_frame, overview_policy_max_frames = _qwen_overview_density_controls_for_allocation(
            config, allocation
        )
        overview_configured_max_frames = min(overview_policy_max_frames, int(total_frames))
        overview_frame_cap = overview_configured_max_frames
        if overview_min_visual_tokens_per_frame is not None:
            overview_frame_cap, overview_density_plan = _qwen_stream_token_density_frame_plan(
                config,
                sample,
                requested_tokens_per_frame=overview_min_visual_tokens_per_frame,
                stream_pixel_budget=int(allocation.overview_pixels),
                total_frames=total_frames,
                configured_max_frames=overview_configured_max_frames,
            )

    local_min_visual_tokens_per_frame = _qwen_local_density_target_for_allocation(config, allocation)
    local_max_frames_override: int | None = None
    local_density_plan: dict[str, Any] | None = None
    if local_min_visual_tokens_per_frame is not None:
        local_max_frames_override, local_density_plan = _qwen_stream_token_density_frame_plan(
            config,
            sample,
            requested_tokens_per_frame=local_min_visual_tokens_per_frame,
            stream_pixel_budget=int(allocation.local_pixels),
            total_frames=total_frames,
            configured_max_frames=_local_sampling_configured_max_frames(config),
        )
    local_indices = _select_region_only_reasoning_indices(
        config,
        local_segments,
        fps=fps,
        total_frames=total_frames,
        local_pixels=int(allocation.local_pixels),
        max_frames_override=local_max_frames_override,
    )
    local_refs = _build_frame_references(local_indices, fps)
    local_frames = _decode_selected_frames(sample, local_indices)

    if overview_frame_cap > 0:
        overview_indices = qwen_fps_frame_indices(total_frames, fps, overview_sampling_fps, overview_frame_cap, min_frames=1)
        overview_refs = _build_frame_references(overview_indices, fps)
        overview_frames = _decode_selected_frames(sample, overview_indices)
    else:
        overview_refs, overview_frames = [], []

    overview_refs = _with_evidence_metadata(
        overview_refs,
        evidence_type="overview",
        evidence_role="global_context_low_resolution",
    )
    local_refs = _with_evidence_metadata(
        local_refs,
        evidence_type="local_detail",
        evidence_role="focused_high_resolution",
        local_segments=local_segments,
    )

    overlap_tolerance_sec = 0.5 / overview_sampling_fps if overview_sampling_fps > 0 else 0.0
    if local_refs and overview_refs:
        kept_overview_pairs = [
            (frame_ref, frame)
            for frame_ref, frame in zip(overview_refs, overview_frames)
            if not any(abs(frame_ref.timestamp_sec - local_ref.timestamp_sec) <= overlap_tolerance_sec for local_ref in local_refs)
        ]
        overview_refs = [frame_ref for frame_ref, _ in kept_overview_pairs]
        overview_frames = [frame for _, frame in kept_overview_pairs]

    allocated_overview_pixels = int(allocation.overview_pixels)
    allocated_local_pixels = int(allocation.local_pixels)
    if not overview_frames:
        allocation.overview_pixels = 0
        allocation.local_pixels = max(allocation.local_pixels, _adaptive_raw_video_total_pixels(config))
    elif not local_frames:
        allocation.overview_pixels = max(allocation.overview_pixels + allocation.local_pixels, allocation.overview_pixels)
        allocation.local_pixels = 0
    if local_frames and allocation.local_pixels > 0:
        local_frames = resize_frames_to_total_pixels(
            local_frames,
            max_total_pixels=int(allocation.local_pixels),
            model_name=config.model.name,
        )
        allocation.metadata["local_budget_estimated_frame_pixels"] = int(_estimated_total_frame_pixels(local_frames))
    final_total_pixels = max(1, int(allocation.overview_pixels) + int(allocation.local_pixels))
    allocation.metadata.update(
        {
            "overview_pixels": int(allocation.overview_pixels),
            "local_pixels": int(allocation.local_pixels),
            "overview_ratio": float(allocation.overview_pixels) / float(final_total_pixels),
            "local_ratio": float(allocation.local_pixels) / float(final_total_pixels),
            "allocated_overview_pixels_before_final_frame_selection": allocated_overview_pixels,
            "allocated_local_pixels_before_final_frame_selection": allocated_local_pixels,
        }
    )

    combined_pairs = [
        *[(frame_ref, "overview") for frame_ref, _ in zip(overview_refs, overview_frames)],
        *[(frame_ref, "segment") for frame_ref, _ in zip(local_refs, local_frames)],
    ]
    combined_pairs.sort(key=lambda item: (item[0].timestamp_sec, item[0].frame_index, 0 if item[1] == "segment" else 1))
    all_refs = [frame_ref for frame_ref, _ in combined_pairs]
    raw_input = _build_hybrid_video_stream_input(
        config=config,
        sample=sample,
        total_frames=total_frames,
        fps=fps,
        overview_refs=overview_refs,
        overview_frames=overview_frames,
        segment_refs=local_refs,
        segment_frames=local_frames,
        overview_pixels=int(allocation.overview_pixels),
        segment_pixels=int(allocation.local_pixels),
        stream_context={
            "local_segment_count": len(local_segments),
            "local_segment_ids": [f"local_segment_{index}" for index, _ in enumerate(local_segments, start=1)],
            "local_segment_sources": [segment.source for segment in local_segments],
            "local_segment_queries": [segment.query for segment in local_segments],
            "local_segment_scores": [float(segment.score) for segment in local_segments],
            "local_segment_frame_indices": [list(segment.frame_indices) for segment in local_segments],
            "local_segment_time_ranges_sec": [[float(segment.start_sec), float(segment.end_sec)] for segment in local_segments],
        },
        qwen_overview_min_visual_tokens_per_frame=overview_min_visual_tokens_per_frame,
        qwen_local_min_visual_tokens_per_frame=local_min_visual_tokens_per_frame,
    )

    metadata: dict[str, Any] = {
        "requested_frame_count": len(all_refs),
        "decoded_frame_count": len(raw_input.frames),
        "hybrid_video_stream_count": len(raw_input.video_streams),
        "hybrid_video_stream_labels": [stream.label for stream in raw_input.video_streams],
        "hybrid_video_stream_kinds": [stream.kind for stream in raw_input.video_streams],
        "overview_frame_count": len(overview_frames),
        "segment_frame_count": len(local_frames),
        "overview_frame_indices": [frame.frame_index for frame in overview_refs],
        "segment_frame_indices": local_indices,
        "overview_total_pixels": int(allocation.overview_pixels),
        "segment_total_pixels": int(allocation.local_pixels),
        "overview_sampling_fps": overview_sampling_fps,
        "overview_sampling_configured_max_frames": int(overview_configured_max_frames),
        **_region_only_reasoning_sampling_metadata(config, fps=fps, local_pixels=int(allocation.local_pixels)),
    }
    if overview_density_plan is not None:
        metadata.update(
            {
                "qwen_overview_token_density_plan": dict(overview_density_plan),
                "overview_sampling_target_tokens_per_frame": int(overview_min_visual_tokens_per_frame or 0),
            }
        )
        allocation.metadata["qwen_overview_min_visual_tokens_per_frame"] = int(overview_min_visual_tokens_per_frame or 0)
    if local_density_plan is not None:
        metadata.update(
            {
                "qwen_local_token_density_plan": dict(local_density_plan),
                "local_sampling_target_tokens_per_frame": int(local_min_visual_tokens_per_frame or 0),
                "local_sampling_policy": "sample_by_fps_then_grid_aware_local_token_density",
            }
        )
        allocation.metadata["qwen_local_min_visual_tokens_per_frame"] = int(local_min_visual_tokens_per_frame or 0)
    return all_refs, raw_input, metadata


# ----------------------------------------------------------------------------------------------------
# Per-question entry point
# ----------------------------------------------------------------------------------------------------


def _result_metadata(
    sample: QuestionSample,
    *,
    retrieval_query_mode: str,
    reasoning_metadata: dict[str, Any] | None,
    frame_metadata: dict[str, Any],
    **fields: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "video_metadata": sample.video_metadata.to_dict(),
        "retrieval_query_mode": retrieval_query_mode,
        "reasoning": reasoning_metadata,
        # Frames sent and visual tokens realized by the final request (read by the accuracy summary).
        "qwen_cache": _augment_qwen_cache_metadata(frame_metadata, reasoning_metadata),
        "retrieval_queries": {"generated": [], "original_question": [], "used": []},
        "query_sanitizer": None,
        "adaptive_plan": None,
        "memory_bank": {"regions": [], "refined_regions": []},
        "refinement": {"enabled": False, "skipped_reason": retrieval_query_mode},
        "allocation": {},
        "fallbacks": [],
        "retrieval_skipped_reason": None,
    }
    metadata.update(fields)
    return metadata


def answer_question_adaalloc(
    config: Config,
    sample: QuestionSample,
    retrieval_backend: UniversalVTGBackend,
    reasoner: Any,
) -> QuestionResult:
    """Answer one question with AdaAlloc.

    The reasoner provides ``generate_overview_plan``, ``refine_memory_bank``, ``answer_question``,
    ``answer_overrides`` and ``last_request_metadata``.
    """

    # A no-op when run.py already prepared the sample.
    sample = prepare_sample(config, sample)
    total_frames = sample.video_metadata.total_frames or config.caps.refiner
    fps = sample.video_metadata.fps or 1.0

    full_coverage_metadata: dict[str, Any] = {}
    direct_answer_inputs = _adaptive_full_coverage_direct_indices(
        config,
        total_frames=total_frames,
        fps=fps,
        sample=sample,
        decision_metadata=full_coverage_metadata,
    )
    if direct_answer_inputs is not None:
        selected_indices, direct_answer_metadata = direct_answer_inputs
        selected_frames, raw_frames, frame_metadata = _prepare_video_path_reasoning_inputs(selected_indices, fps=fps)
        prediction = _answer_question_with_prompt_style(
            reasoner,
            config,
            sample,
            selected_frames,
            raw_frames,
            [],
            use_video_path_preprocessing=True,
            qwen_video_path_frame_plan=direct_answer_metadata.get("qwen_direct_frame_plan"),
        )
        direct_answer_metadata.update({"answer_prompt_style": SIMPLE_MCQ_PROMPT_STYLE, "qwen_video_path_preprocessing": True})
        reasoning_metadata = getattr(reasoner, "last_request_metadata", None)
        _augment_full_coverage_processor_density(direct_answer_metadata, reasoning_metadata)
        return QuestionResult(
            sample=sample,
            selected_frames=selected_frames,
            retrieved_segments=[],
            generated_queries=[],
            prediction=prediction,
            metadata=_result_metadata(
                sample,
                retrieval_query_mode="adaptive_full_coverage_direct_answer",
                reasoning_metadata=reasoning_metadata,
                frame_metadata=frame_metadata,
                retrieval_skipped_reason="adaptive_full_coverage_direct_answer",
                adaptive_full_coverage_direct_answer=direct_answer_metadata,
            ),
        )

    # 1. Planner: allocation policy and retrieval queries from a sparse overview.
    fallback_reasons: list[str] = []
    overview_refs, overview_raw_frames, overview_frame_metadata = _prepare_adaptive_overview_inputs(
        config,
        sample,
        total_frames=total_frames,
        fps=fps,
    )
    plan = reasoner.generate_overview_plan(sample, overview_refs, overview_raw_frames)
    planner_reasoning_metadata = getattr(reasoner, "last_request_metadata", None)
    overview_frame_metadata = _augment_qwen_cache_metadata(overview_frame_metadata, planner_reasoning_metadata)
    if isinstance(planner_reasoning_metadata, dict):
        overview_frame_metadata["planner_reasoning"] = dict(planner_reasoning_metadata)
    if plan.parse_failed or plan.fallback_reason:
        fallback_reasons.append(plan.fallback_reason or "overview_plan_parse_failed")
    if plan.parse_failed and str(plan.fallback_reason or "").startswith("parse_failure:"):
        selected_indices, parse_failure_answer_metadata = _adaptive_plan_parse_failure_uniform_indices(
            config,
            total_frames=total_frames,
        )
        selected_frames, raw_frames, frame_metadata = _prepare_reasoning_inputs(sample, selected_indices, fps=fps)
        prediction = _answer_question_with_prompt_style(reasoner, config, sample, selected_frames, raw_frames, [])
        parse_failure_answer_metadata["fallback_reason"] = plan.fallback_reason
        return QuestionResult(
            sample=sample,
            selected_frames=selected_frames,
            retrieved_segments=[],
            generated_queries=[],
            prediction=prediction,
            metadata=_result_metadata(
                sample,
                retrieval_query_mode="adaptive_plan_parse_failure_uniform_simple_mcq",
                reasoning_metadata=getattr(reasoner, "last_request_metadata", None),
                frame_metadata=frame_metadata,
                adaptive_plan=plan.to_dict(),
                fallbacks=fallback_reasons,
                overview_planning_qwen_cache=overview_frame_metadata,
                retrieval_skipped_reason="adaptive_plan_parse_failure_uniform_simple_mcq",
                adaptive_plan_parse_failure_uniform_answer=parse_failure_answer_metadata,
                adaptive_full_coverage_direct_answer=full_coverage_metadata or None,
            ),
        )
    plan = _apply_question_time_interval_override(sample, plan)
    fixed_policy_metadata = None if config.adaptive_allocation else force_allocation_policy(plan, config.fixed_policy)

    # 2. Retrieval, gated by confidence.
    retrieved_segments: list[RetrievedSegment] = []
    retrieval_skipped_reason = None
    direct_answer = False
    confidence_gate_metadata: dict[str, Any] | None = None
    local_retrieval_needed = not config.adaptive_allocation or (
        bool(plan.local_retrieval_needed)
        if plan.local_retrieval_needed is not None
        else bool(plan.budget_policy.local_evidence_needed)
    )
    generated_retrieval_queries, question_retrieval_queries, retrieval_queries_used, query_sanitizer_metadata = (
        _adaptive_retrieval_query_sets(sample, plan)
    )
    if local_retrieval_needed and retrieval_queries_used:
        retrieved_segments = retrieval_backend.retrieve(
            sample=sample,
            queries=retrieval_queries_used,
            budget=config.caps.refiner,
        )
        retrieved_segments = _mark_question_retrieval_segments(retrieved_segments, question_retrieval_queries)
        retrieved_segments, direct_answer, confidence_gate_metadata, gate_fallbacks = (
            _apply_adaptive_retrieval_confidence_gate(config, retrieved_segments)
        )
        fallback_reasons.extend(gate_fallbacks)
        if not retrieved_segments:
            fallback_reasons.append("empty_retrieval")
    else:
        if not local_retrieval_needed:
            question_retrieval_queries = []
            retrieval_queries_used = []
        retrieval_skipped_reason = (
            "overview_plan_no_universal_vtg_retrieval"
            if plan.budget_policy.local_evidence_needed
            else "overview_plan_no_local_evidence"
        )
        fallback_reasons.append(retrieval_skipped_reason)

    # 3. Memory bank and refinement.
    memory_bank = build_memory_bank(sample, plan, retrieved_segments, config)
    refined_regions = list(memory_bank.regions)
    refinement_metadata: dict[str, Any] = {"enabled": True, "skipped_reason": None}
    refinement: AdaptiveMemoryRefinement
    local_evidence_needed = bool(plan.budget_policy.local_evidence_needed)
    if memory_bank.regions and local_evidence_needed and not direct_answer:
        candidate_indices = sample_candidate_frame_indices(
            memory_bank.regions,
            native_fps=fps,
            total_frames=total_frames,
            config=config,
        )
        candidate_refs, candidate_raw_frames, candidate_frame_metadata = _prepare_reasoning_inputs(
            sample,
            candidate_indices,
            fps=fps,
        )
        snap_boundary_frames = bool(config.refiner.snap_boundary_frames and fps)
        candidate_frames_by_region = candidate_frame_indices_by_region(
            memory_bank.regions,
            candidate_refs,
            tolerance_sec=0.5 / float(fps) if snap_boundary_frames else 0.0,
        )
        candidate_counts_by_region = {region_id: len(indices) for region_id, indices in candidate_frames_by_region.items()}
        refinement_raw_frames = _build_refinement_video_stream_input(
            config,
            sample=sample,
            memory_regions=memory_bank.regions,
            candidate_refs=candidate_refs,
            candidate_frames=candidate_raw_frames,
            total_frames=total_frames,
            fps=fps,
            region_frame_indices=candidate_frames_by_region if snap_boundary_frames else None,
        )
        refinement_context: dict[str, Any] = {"plan": plan, "frame_counts_by_region": candidate_counts_by_region}
        if snap_boundary_frames:
            refinement_context["context_clip_count"] = sum(
                stream.metadata.get("segment_id") == "segment_unmatched" for stream in refinement_raw_frames.video_streams
            )
        refinement = reasoner.refine_memory_bank(
            sample,
            memory_bank.regions,
            candidate_refs,
            refinement_raw_frames,
            retrieved_segments,
            refinement_context=refinement_context,
        )
        refinement_reasoning_metadata = getattr(reasoner, "last_request_metadata", None)
        if refinement.parse_failed or refinement.fallback_reason:
            fallback_reasons.append(refinement.fallback_reason or "refinement_parse_failed")
        refined_regions, refinement_apply_metadata = apply_refinement(
            memory_bank.regions,
            refinement,
            duration_sec=sample.video_metadata.duration_sec,
            min_remove_confidence=config.refiner.min_remove_confidence,
            max_uncertainty_for_removal=config.refiner.max_uncertainty_for_removal,
            keep_score_strength=config.refiner.keep_score_strength,
            min_keep_confidence=config.refiner.min_keep_confidence,
            min_adjust_confidence=config.refiner.min_adjust_confidence,
            max_focus_regions=config.refiner.max_focus_regions,
            frame_counts_by_region=candidate_counts_by_region,
            min_remove_frames=config.refiner.min_remove_frames,
            min_merge_confidence=config.refiner.min_merge_confidence,
            max_merge_gap_sec=config.refiner.max_merge_gap_sec,
            block_removes=bool(
                config.refiner.block_removes_for_order_count and is_order_or_count_question(sample.question_text)
            ),
        )
        refinement_metadata.update(
            {
                "candidate_frames": candidate_frame_metadata,
                "candidate_frame_indices": [frame.frame_index for frame in candidate_refs],
                "candidate_sampling": {
                    "sample_fps": float(config.fps.refiner),
                    "max_frames": int(config.caps.refiner),
                    "selected_frame_count": len(candidate_refs),
                    "video_stream_count": len(refinement_raw_frames.video_streams),
                    "video_stream_region_ids": [
                        str(stream.metadata.get("memory_region_id") or "") for stream in refinement_raw_frames.video_streams
                    ],
                    "frame_counts_by_region": candidate_counts_by_region,
                },
                "reasoning": (
                    dict(refinement_reasoning_metadata)
                    if isinstance(refinement_reasoning_metadata, dict)
                    else refinement_reasoning_metadata
                ),
                "result": refinement.to_dict(),
                "apply": refinement_apply_metadata,
            }
        )
        if (
            config.refiner.no_evidence_fallback
            and refinement.usable_local_evidence is False
            and not refinement.parse_failed
            and float(refinement.uncertainty) <= float(config.refiner.max_uncertainty_for_removal)
            and not any(action.helpful in {"yes", "partly"} for action in refinement.actions)
        ):
            # The refiner saw every candidate clip and found nothing that helps answer: answer from
            # uniform frames instead, through the same path as the weak-retrieval fallback.
            direct_answer = True
            fallback_reasons.append("refinement_no_usable_local_evidence")
            refinement_metadata["no_evidence_fallback"] = True
    else:
        if direct_answer:
            skip_reason = "filtered_weak_retrieval_direct_answer"
        elif not local_evidence_needed:
            skip_reason = "overview_plan_no_local_evidence"
        else:
            skip_reason = "empty_memory_bank"
        refinement = fallback_memory_refinement(reason=skip_reason)
        refinement_metadata.update({"skipped_reason": skip_reason, "result": refinement.to_dict()})

    # 4. Allocation and final answer.
    local_segments = (
        [] if direct_answer else regions_to_retrieved_segments(refined_regions) if local_evidence_needed else []
    )
    allocation = allocate_visual_budget(
        plan,
        uncertainty=max(float(plan.budget_policy.uncertainty), float(refinement.uncertainty)),
        has_local_evidence=bool(local_segments),
        total_pixel_budget=_adaptive_raw_video_total_pixels(config),
        fallback_reasons=fallback_reasons,
    )
    allocation.metadata["planner_policy_name"] = (
        fixed_policy_metadata["planner_policy_name"] if fixed_policy_metadata else plan.budget_policy.policy_name
    )
    allocation.metadata["adaptive_allocation"] = bool(config.adaptive_allocation)
    if fixed_policy_metadata is not None:
        allocation.metadata["fixed_policy"] = fixed_policy_metadata
    weak_retrieval_direct_answer_metadata: dict[str, Any] | None = None
    if direct_answer:
        direct_answer_inputs = _adaptive_full_coverage_direct_indices(
            config,
            total_frames=total_frames,
            fps=fps,
            sample=sample,
            force_direct=True,
        )
        if direct_answer_inputs is None:
            raise RuntimeError("The weak-retrieval and no-evidence fallbacks found no usable uniform frames.")
        selected_indices, weak_retrieval_direct_answer_metadata = direct_answer_inputs
        selected_frames, raw_frames, frame_metadata = _prepare_video_path_reasoning_inputs(selected_indices, fps=fps)
        prediction = _answer_question_with_prompt_style(
            reasoner,
            config,
            sample,
            selected_frames,
            raw_frames,
            [],
            use_video_path_preprocessing=True,
            qwen_video_path_frame_plan=weak_retrieval_direct_answer_metadata.get("qwen_direct_frame_plan"),
        )
        weak_retrieval_direct_answer_metadata.update(
            {
                "answer_prompt_style": SIMPLE_MCQ_PROMPT_STYLE,
                "qwen_video_path_preprocessing": True,
                "retrieval_confidence_gate": confidence_gate_metadata,
                "refinement_no_evidence_fallback": bool(refinement_metadata.get("no_evidence_fallback")),
            }
        )
        allocation.metadata.update(
            {
                "used_for_final_answer": False,
                "filtered_weak_retrieval_direct_answer": True,
                "weak_retrieval_trigger_reason": (confidence_gate_metadata or {}).get("trigger_reason"),
            }
        )
    else:
        selected_frames, raw_frames, frame_metadata = _prepare_adaptive_final_reasoning_inputs(
            config,
            sample,
            local_segments,
            allocation,
            total_frames=total_frames,
            fps=fps,
        )
        prediction = reasoner.answer_question(sample, selected_frames, raw_frames, local_segments)
    reasoning_metadata = getattr(reasoner, "last_request_metadata", None)

    return QuestionResult(
        sample=sample,
        selected_frames=selected_frames,
        retrieved_segments=retrieved_segments,
        generated_queries=plan.queries,
        prediction=prediction,
        metadata=_result_metadata(
            sample,
            retrieval_query_mode="query_adaptive_memory_bank",
            reasoning_metadata=reasoning_metadata,
            frame_metadata=frame_metadata,
            retrieval_queries={
                "generated": generated_retrieval_queries,
                "original_question": question_retrieval_queries,
                "used": retrieval_queries_used,
            },
            query_sanitizer=query_sanitizer_metadata,
            adaptive_plan=plan.to_dict(),
            memory_bank={**memory_bank.to_dict(), "refined_regions": [region.to_dict() for region in refined_regions]},
            refinement=refinement_metadata,
            allocation=allocation.metadata,
            fallbacks=fallback_reasons,
            overview_planning_qwen_cache=overview_frame_metadata,
            retrieval_skipped_reason=retrieval_skipped_reason,
            adaptive_retrieval_confidence_gate=confidence_gate_metadata,
            adaptive_filtered_weak_retrieval_direct_answer=weak_retrieval_direct_answer_metadata,
            adaptive_full_coverage_direct_answer=full_coverage_metadata or None,
        ),
    )
