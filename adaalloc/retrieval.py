"""UniversalVTG temporal grounding over PE-Core features, and frame sampling from retrieved segments."""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import Config
from .data import RetrievedSegment
from .video import probe_video_metadata


def clamp_segment_to_duration(start_sec: float, end_sec: float, duration_sec: float | None) -> tuple[float, float]:
    start = max(0.0, float(start_sec))
    end = max(start, float(end_sec))
    if duration_sec is not None:
        duration = max(0.0, float(duration_sec))
        start = min(start, duration)
        end = min(max(end, start), duration)
    return start, end


def _downsample_preserving_order(indices: Sequence[int], budget: int) -> list[int]:
    if budget <= 0 or not indices:
        return []
    if budget >= len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, num=budget)
    return np.unique(np.asarray(indices)[np.round(positions).astype(int)]).astype(int).tolist()


def _sample_segment_indices_at_fps(
    *,
    start_sec: float,
    end_sec: float,
    native_fps: float,
    sample_fps: float,
    total_frames: int | None,
) -> list[int]:
    max_duration = (total_frames - 1) / native_fps if total_frames is not None and total_frames > 0 else None
    start_sec, end_sec = clamp_segment_to_duration(start_sec, end_sec, max_duration)
    timestamps = np.arange(start_sec, end_sec + 1e-9, 1.0 / sample_fps)
    if timestamps.size == 0:
        timestamps = np.asarray([(start_sec + end_sec) / 2.0], dtype=np.float64)

    indices = np.round(timestamps * native_fps).astype(int).tolist()
    if total_frames is not None and total_frames > 0:
        indices = [min(max(0, index), total_frames - 1) for index in indices]
    return sorted(set(indices))


def _sample_segment_indices_at_fps_with_expansion(
    *,
    start_sec: float,
    end_sec: float,
    native_fps: float,
    sample_fps: float,
    total_frames: int | None,
    max_frames: int,
) -> list[int]:
    indices = _sample_segment_indices_at_fps(
        start_sec=start_sec,
        end_sec=end_sec,
        native_fps=native_fps,
        sample_fps=sample_fps,
        total_frames=total_frames,
    )
    if (
        max_frames <= 0
        or len(indices) >= max_frames
        or total_frames is None
        or total_frames <= 0
        or native_fps <= 0
        or sample_fps <= 0
    ):
        return _downsample_preserving_order(indices, max_frames)

    max_duration = (total_frames - 1) / native_fps
    center_sec = min(max((float(start_sec) + float(end_sec)) / 2.0, 0.0), max_duration)
    max_available_frames = min(max_frames, max(1, int(math.floor(max_duration * sample_fps + 1e-9)) + 1))
    if max_available_frames == 1:
        expanded_timestamps = np.asarray([center_sec], dtype=np.float64)
    else:
        total_span = (max_available_frames - 1) / sample_fps
        expanded_start = center_sec - total_span / 2.0
        expanded_end = expanded_start + total_span
        if expanded_start < 0:
            expanded_end += -expanded_start
            expanded_start = 0.0
        if expanded_end > max_duration:
            shift = expanded_end - max_duration
            expanded_start = max(0.0, expanded_start - shift)
        expanded_timestamps = expanded_start + np.arange(max_available_frames, dtype=np.float64) / sample_fps

    expanded_indices = np.round(expanded_timestamps * native_fps).astype(int).tolist()
    expanded_indices = [min(max(0, index), total_frames - 1) for index in expanded_indices]
    return sorted(set(expanded_indices))


def segment_to_frame_indices(*, start_sec: float, end_sec: float, fps: float, total_frames: int | None, max_frames: int) -> list[int]:
    """Up to ``max_frames`` native frame indices spread over a segment."""
    if max_frames <= 0 or fps <= 0:
        return []

    max_duration = None
    if total_frames is not None and total_frames > 0:
        max_duration = (total_frames - 1) / fps
    start_sec, end_sec = clamp_segment_to_duration(start_sec, end_sec, max_duration)

    start_idx = max(0, int(math.ceil(start_sec * fps - 1e-9)))
    end_idx = max(start_idx, int(math.floor(end_sec * fps + 1e-9)))
    if total_frames is not None:
        end_idx = min(end_idx, max(0, total_frames - 1))
        start_idx = min(start_idx, end_idx)

    candidate = list(range(start_idx, end_idx + 1))
    if not candidate:
        fallback = max(0, int(round(((start_sec + end_sec) / 2.0) * fps)))
        if total_frames is not None:
            fallback = min(fallback, max(0, total_frames - 1))
        candidate = [fallback]

    return _downsample_preserving_order(candidate, max_frames)


def allocate_segment_frame_budgets(scores: list[float], total_budget: int, temperature: float = 0.2) -> list[int]:
    """Split ``total_budget`` frames across segments by a softmax over their scores."""
    if total_budget <= 0 or not scores:
        return [0 for _ in scores]
    if len(scores) == 1:
        return [total_budget]

    logits = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-6)
    logits -= logits.max(initial=0.0)
    weights = np.exp(logits)
    if not np.isfinite(weights).all() or float(weights.sum()) == 0.0:
        weights = np.ones_like(weights)
    weights /= weights.sum()

    raw = weights * total_budget
    allocation = np.floor(raw).astype(int)
    remainder = int(total_budget - allocation.sum())
    if remainder > 0:
        fractional = raw - allocation
        for idx in np.argsort(-fractional, kind="stable")[:remainder]:
            allocation[idx] += 1
    return allocation.tolist()


def _segment_field(segment: Any, name: str) -> float:
    return float(getattr(segment, name))


def sample_frames_from_retrieved_segments(
    segments: Sequence[Any],
    *,
    native_fps: float,
    sample_fps: float,
    total_frames: int | None,
    max_frames: int,
    sampling_mode: str = "temporal_union",
    min_frames_per_region: int = 8,
) -> list[int]:
    """Sample native frame indices at ``sample_fps`` inside the segments, at most ``max_frames`` in total.

    ``temporal_union`` (final local stream) splits the frames across segments by score and expands a
    short segment around its center to use its share; ``region_length`` (refiner candidates) splits
    them by segment length with a per-region floor of ``min_frames_per_region``.
    """
    if max_frames <= 0 or native_fps <= 0 or sample_fps <= 0 or not segments:
        return []
    if sampling_mode not in {"temporal_union", "region_length"}:
        raise ValueError(f"sampling_mode must be 'temporal_union' or 'region_length'; got {sampling_mode!r}.")

    if sampling_mode == "region_length":
        segment_indices = [
            _sample_segment_indices_at_fps(
                start_sec=_segment_field(segment, "start_sec"),
                end_sec=_segment_field(segment, "end_sec"),
                native_fps=native_fps,
                sample_fps=sample_fps,
                total_frames=total_frames,
            )
            for segment in segments
        ]
        # Frames follow region length, so every region is sampled about equally densely, except that a region
        # whose share would fall below the floor gets the floor (or all its frames) so it can still be judged.
        non_empty = [indices for indices in segment_indices if indices]
        if not non_empty:
            return []
        combined = sorted(set(index for indices in non_empty for index in indices))
        if len(combined) <= max_frames:
            return combined
        lengths = [len(indices) for indices in non_empty]
        floors = [min(length, max(0, int(min_frames_per_region))) for length in lengths]
        if sum(floors) > max_frames:
            quotas = [min(length, max_frames // len(non_empty)) for length in lengths]
        else:
            quotas: list[int | None] = [None] * len(lengths)
            budget = max_frames
            free = list(range(len(lengths)))
            while free:
                total = sum(lengths[i] for i in free)
                shares = {i: budget * lengths[i] / total for i in free}
                fixed = [i for i in free if shares[i] < floors[i] or shares[i] > lengths[i]]
                if not fixed:
                    break
                for i in fixed:
                    quotas[i] = floors[i] if shares[i] < floors[i] else lengths[i]
                    budget -= quotas[i]
                free = [i for i in free if quotas[i] is None]
            if free:
                total = sum(lengths[i] for i in free)
                raw = {i: budget * lengths[i] / total for i in free}
                for i in free:
                    quotas[i] = int(raw[i])
                leftover = budget - sum(int(raw[i]) for i in free)
                for i in sorted(free, key=lambda i: raw[i] - int(raw[i]), reverse=True)[:leftover]:
                    quotas[i] += 1
        selected = set()
        for indices, quota in zip(non_empty, quotas):
            selected.update(_downsample_preserving_order(indices, quota))
        if len(selected) < max_frames:
            rest = [index for index in combined if index not in selected]
            selected.update(_downsample_preserving_order(rest, max_frames - len(selected)))
        return _downsample_preserving_order(sorted(selected), max_frames)

    budgets = allocate_segment_frame_budgets(
        [_segment_field(segment, "score") for segment in segments],
        max_frames,
    )
    weighted_indices: list[int] = []
    for segment, budget in zip(segments, budgets):
        weighted_indices.extend(
            _sample_segment_indices_at_fps_with_expansion(
                start_sec=_segment_field(segment, "start_sec"),
                end_sec=_segment_field(segment, "end_sec"),
                native_fps=native_fps,
                sample_fps=sample_fps,
                total_frames=total_frames,
                max_frames=budget,
            )
        )
    return sorted(set(weighted_indices))


def _resolve_universal_vtg_root(configured_root: str | None) -> Path:
    """The configured root, else $ADAALLOC_UNIVERSAL_VTG_ROOT, else the ``third_party/universalvtg`` submodule."""
    root = configured_root or os.environ.get("ADAALLOC_UNIVERSAL_VTG_ROOT")
    if root:
        return Path(root).expanduser()
    return Path(__file__).resolve().parents[1] / "third_party" / "universalvtg"


@dataclass(slots=True, frozen=True)
class PredictedSegment:
    start_sec: float
    end_sec: float
    score: float
    query_index: int
    rank: int


@dataclass(slots=True, frozen=True)
class VTGQueryPrediction:
    query: Any
    segments: tuple[PredictedSegment, ...]


@dataclass(slots=True, frozen=True)
class UniversalVTGPrediction:
    queries: tuple[Any, ...]
    predictions: tuple[VTGQueryPrediction, ...]


def load_pecore_features(features_or_path: Any) -> torch.Tensor:
    """Load PE-Core features as a (1, 1024, T) float tensor."""
    if isinstance(features_or_path, (str, Path)):
        try:
            loaded = torch.load(features_or_path, map_location="cpu", weights_only=True)
        except Exception:
            loaded = torch.load(features_or_path, map_location="cpu")
    else:
        loaded = features_or_path

    if isinstance(loaded, dict):
        loaded = loaded.get("features", loaded.get("visual_features", loaded))

    if not isinstance(loaded, torch.Tensor):
        loaded = torch.as_tensor(loaded)

    if loaded.ndim != 2:
        raise ValueError(f"Expected PECore tensor shaped (T, 1024) or (1024, T), got {tuple(loaded.shape)}")

    loaded = loaded.detach().cpu().to(torch.float32)
    if loaded.shape[-1] == 1024:
        loaded = loaded.transpose(0, 1)
    elif loaded.shape[0] != 1024:
        raise ValueError(f"Expected PECore tensor shaped (T, 1024) or (1024, T), got {tuple(loaded.shape)}")

    return loaded.unsqueeze(0).contiguous()


def mean_pool_features_in_time(features: torch.Tensor, stride: int) -> torch.Tensor:
    """Average each run of ``stride`` time steps of a (..., D, T) tensor; a short tail is averaged on its own."""
    if stride <= 1:
        return features
    length = features.size(-1)
    full = length // stride * stride
    pooled = [features[..., :full].reshape(*features.shape[:-1], full // stride, stride).mean(dim=-1)]
    if full < length:
        pooled.append(features[..., full:].mean(dim=-1, keepdim=True))
    return torch.cat(pooled, dim=-1).contiguous()


def fit_stride_for_point_buffer(length: int, *, max_points: int, padding_stride: int) -> int:
    """Smallest integer pooling stride whose padded length fits UniversalVTG's point buffer."""
    padding_stride = max(1, padding_stride)

    def padded_length(steps: int) -> int:
        return -(-steps // padding_stride) * padding_stride

    stride = 1
    while padded_length(-(-length // stride)) > max_points:
        stride += 1
    return stride


def _normalize_queries(query: Any) -> tuple[Any, ...]:
    if isinstance(query, (list, tuple)):
        return tuple(query)
    return (query,)


def _normalize_prediction(results: dict[str, Any], queries: Sequence[Any], duration_seconds: float) -> UniversalVTGPrediction:
    raw_segments = results.get("segments", [])
    raw_scores = results.get("scores", [])

    if isinstance(raw_segments, torch.Tensor):
        segment_batches = [batch for batch in raw_segments] if raw_segments.ndim == 3 else [raw_segments]
    else:
        segment_batches = [torch.as_tensor(batch, dtype=torch.float32) for batch in raw_segments]

    if isinstance(raw_scores, torch.Tensor):
        score_batches = [batch for batch in raw_scores] if raw_scores.ndim == 2 else [raw_scores]
    else:
        score_batches = [torch.as_tensor(batch, dtype=torch.float32) for batch in raw_scores]

    if len(segment_batches) != len(queries) or len(score_batches) != len(queries):
        raise ValueError(
            "UniversalVTG returned mismatched batch shapes: "
            f"segments={len(segment_batches)}, scores={len(score_batches)}, queries={len(queries)}"
        )

    predictions: list[VTGQueryPrediction] = []
    for query_index, query in enumerate(queries):
        normalized_segments: list[PredictedSegment] = []
        for rank, (segment, score) in enumerate(zip(segment_batches[query_index], score_batches[query_index])):
            start_sec, end_sec = clamp_segment_to_duration(float(segment[0].item()), float(segment[1].item()), duration_seconds)
            normalized_segments.append(
                PredictedSegment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    score=float(score.item()),
                    query_index=query_index,
                    rank=rank,
                )
            )
        predictions.append(VTGQueryPrediction(query=query, segments=tuple(normalized_segments)))
    return UniversalVTGPrediction(queries=tuple(queries), predictions=tuple(predictions))


class UniversalVTGBackend:
    """Grounds text queries in a video with UniversalVTG over the dataset's cached PE-Core features."""

    def __init__(self, config: Config) -> None:
        self.retrieval_config = config.retrieval
        self.pecore_feature_dir = config.dataset.pecore_features if config.dataset is not None else None
        root = _resolve_universal_vtg_root(config.retrieval.universal_vtg_root)
        self.module_path = root / "universal_vtg_inference.py"
        self.nms_setup_path = root / "libs" / "nms" / "setup_nms.py"
        self.experiment_dir = root / config.retrieval.experiment
        self._runtime_module = None
        self._model = None
        self._attempted_nms_rebuild = False

    @staticmethod
    def _ensure_mamba_transformers_compatibility() -> None:
        """Bridge older mamba_ssm imports against newer Transformers generation API."""
        try:
            import transformers.generation as generation
            from transformers.generation.utils import GenerateDecoderOnlyOutput
        except Exception:
            return

        if not hasattr(generation, "GreedySearchDecoderOnlyOutput"):
            generation.GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
        if not hasattr(generation, "SampleDecoderOnlyOutput"):
            generation.SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput

    def _ensure_runtime(self):
        if self._runtime_module is not None:
            return self._runtime_module
        self._ensure_mamba_transformers_compatibility()
        runtime_root_str = str(self.module_path.parent)
        if runtime_root_str not in sys.path:
            sys.path.insert(0, runtime_root_str)
        spec = importlib.util.spec_from_file_location("adaalloc_universal_vtg_runtime", self.module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load UniversalVTG runtime from {self.module_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            if self._should_rebuild_nms(exc):
                self._rebuild_nms_extension()
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                raise
        self._runtime_module = module
        return module

    def _should_rebuild_nms(self, exc: Exception) -> bool:
        if self._attempted_nms_rebuild:
            return False
        message = str(exc)
        return "nms_1d_cpu_vg" in message or "undefined symbol" in message

    def _rebuild_nms_extension(self) -> None:
        if self._attempted_nms_rebuild:
            return
        self._attempted_nms_rebuild = True
        if not self.nms_setup_path.exists():
            raise FileNotFoundError(f"Unable to rebuild UniversalVTG NMS extension because {self.nms_setup_path} is missing.")
        subprocess.run(
            # --force: a build/ directory left from another PyTorch would otherwise be re-copied as is.
            [sys.executable, str(self.nms_setup_path), "build_ext", "--inplace", "--force"],
            cwd=str(self.nms_setup_path.parent),
            check=True,
        )

    @property
    def model(self):
        if self._model is None:
            runtime = self._ensure_runtime()
            self._model = runtime.UniversalVTG(
                experiment_name=str(self.experiment_dir),
                checkpoint_name=self.retrieval_config.checkpoint,
                device=self.retrieval_config.device,
            )
        return self._model

    @staticmethod
    def _batch_encoded_queries(encoded_queries: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not encoded_queries:
            raise ValueError("Cannot batch an empty UniversalVTG query list.")
        max_length = max(int(encoded.shape[-1]) for encoded in encoded_queries)
        first = encoded_queries[0]
        batched = first.new_zeros((len(encoded_queries), int(first.shape[0]), max_length))
        mask = torch.zeros((len(encoded_queries), max_length), dtype=torch.bool, device=first.device)
        for query_index, encoded in enumerate(encoded_queries):
            if encoded.ndim != 2:
                raise ValueError(f"Expected encoded query tensor shaped (C, L), got {tuple(encoded.shape)}")
            length = int(encoded.shape[-1])
            if int(encoded.shape[0]) != int(first.shape[0]):
                raise ValueError(
                    "UniversalVTG encoded query channel mismatch: "
                    f"expected {int(first.shape[0])}, got {int(encoded.shape[0])}"
                )
            batched[query_index, :, :length] = encoded.to(device=first.device, dtype=first.dtype)
            mask[query_index, :length] = True
        return batched, mask

    def _resolve_feature_path(self, video_id: str) -> Path:
        root = Path(self.pecore_feature_dir or "")
        for candidate in (root / f"{video_id}.pt", root / video_id / "features.pt"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No PECore feature file found for {video_id} under {root}")

    def predict(
        self,
        features_or_path: Any,
        query: Any,
        *,
        duration_seconds: float,
        top_k: int | None = None,
    ) -> UniversalVTGPrediction:
        """Top-``top_k`` segments per query, in seconds."""
        features = load_pecore_features(features_or_path)
        queries = _normalize_queries(query)
        if not queries:
            return UniversalVTGPrediction(queries=(), predictions=())
        # Cached PE-Core tensors are sampled at retrieval.feature_fps, so timestamps are derived from
        # feature time; the runtime's source-fps conversion (fps=None here) is not used.
        prediction_fps = None
        prediction_feature_fps = float(self.retrieval_config.feature_fps)
        prediction_top_k = int(self.retrieval_config.top_k if top_k is None else top_k)
        # The model's point generator holds at most pt_gen.max_seq_len points, so a very long video
        # (more than ~5 hours at 2 FPS) would fail an assertion. Mean-pool such inputs over time and
        # divide feature_fps by the same stride, so predicted segments still map to correct seconds.
        max_points = int(((getattr(self.model, "opt", None) or {}).get("pt_gen") or {}).get("max_seq_len") or 0)
        if max_points > 0:
            pool_stride = fit_stride_for_point_buffer(
                features.size(-1),
                max_points=max_points,
                padding_stride=int(getattr(self.model, "padding_stride", 1) or 1),
            )
            if pool_stride > 1:
                print(
                    f"[UniversalVTG] {features.size(-1)} feature steps exceed the {max_points}-point buffer; "
                    f"mean-pooling by {pool_stride} (feature_fps {prediction_feature_fps:g} -> "
                    f"{prediction_feature_fps / pool_stride:g}).",
                    file=sys.stderr,
                )
                features = mean_pool_features_in_time(features, pool_stride)
                prediction_feature_fps /= pool_stride
        if len(queries) == 1:
            results = self.model.predict(
                features,
                queries[0],
                fps=prediction_fps,
                feature_fps=prediction_feature_fps,
                duration=float(duration_seconds),
                top_k=prediction_top_k,
            )
            return _normalize_prediction(results, queries, float(duration_seconds))

        encode_text = getattr(self.model, "encode_text", None)
        if callable(encode_text):
            encoded_queries = [
                torch.as_tensor(encode_text(single_query), dtype=features.dtype, device=features.device)
                for single_query in queries
            ]
            batched_text, text_mask = self._batch_encoded_queries(encoded_queries)
            batched_features = features.expand(len(queries), -1, -1).contiguous()
            results = self.model.predict(
                batched_features,
                batched_text,
                fps=prediction_fps,
                feature_fps=prediction_feature_fps,
                duration=float(duration_seconds),
                top_k=prediction_top_k,
                text_mask=text_mask,
            )
            return _normalize_prediction(results, queries, float(duration_seconds))

        combined_predictions: list[VTGQueryPrediction] = []
        for query_index, single_query in enumerate(queries):
            results = self.model.predict(
                features,
                single_query,
                fps=prediction_fps,
                feature_fps=prediction_feature_fps,
                duration=float(duration_seconds),
                top_k=prediction_top_k,
            )
            normalized = _normalize_prediction(results, (single_query,), float(duration_seconds))
            query_prediction = normalized.predictions[0]
            combined_segments = tuple(
                PredictedSegment(
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    score=segment.score,
                    query_index=query_index,
                    rank=segment.rank,
                )
                for segment in query_prediction.segments
            )
            combined_predictions.append(
                VTGQueryPrediction(
                    query=single_query,
                    segments=combined_segments,
                )
            )
        return UniversalVTGPrediction(queries=tuple(queries), predictions=tuple(combined_predictions))

    def retrieve(self, *, sample, queries: list[str], budget: int) -> list[RetrievedSegment]:
        """Retrieve the top segments of every query (no NMS) and spread ``budget`` frames across them by score."""
        if not queries or budget <= 0:
            return []

        metadata = sample.video_metadata
        if metadata.total_frames is None or metadata.fps is None or metadata.duration_sec is None:
            probed = probe_video_metadata(sample.video_path, fallback_fps=metadata.fps)
            if metadata.total_frames is None:
                metadata.total_frames = probed.total_frames
            if metadata.fps is None:
                metadata.fps = probed.fps
            if metadata.duration_sec is None:
                metadata.duration_sec = probed.duration_sec
            if metadata.source == "unknown":
                metadata.source = probed.source

        duration = metadata.duration_sec
        video_fps = metadata.fps or self.retrieval_config.feature_fps
        total_frames = metadata.total_frames
        if duration is None and total_frames and video_fps:
            duration = total_frames / video_fps
        duration = float(duration or 0.0)

        prediction = self.predict(
            self._resolve_feature_path(sample.video_id),
            queries,
            duration_seconds=duration,
            top_k=self.retrieval_config.top_k,
        )

        flattened = [segment for query_prediction in prediction.predictions for segment in query_prediction.segments]
        allocations = allocate_segment_frame_budgets([segment.score for segment in flattened], budget)

        outputs: list[RetrievedSegment] = []
        for segment, segment_budget in zip(flattened, allocations):
            if segment_budget <= 0:
                continue
            frame_indices = segment_to_frame_indices(
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                fps=video_fps,
                total_frames=total_frames,
                max_frames=segment_budget,
            )
            outputs.append(
                RetrievedSegment(
                    query=queries[segment.query_index],
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    score=segment.score,
                    frame_indices=frame_indices,
                    source="universal_vtg",
                )
            )
        return outputs
