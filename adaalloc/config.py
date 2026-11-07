"""AdaAlloc configuration.

``budget.num_frames`` is the only budget knob: every MLLM call gets ``num_frames x 256`` visual
tokens, the cost of sampling that many frames at 256 tokens per frame. 128 frames is 32,768 tokens
(the paper's 32K setting) and 64 frames is 16,384 (16K).

Frame caps are written as multiples of ``num_frames`` (``frames.*``), so they scale with the budget
(half as many frames at 64, each at the same resolution). Per-frame resolution floors, sampling rates,
retrieval, memory and refinement settings do not depend on the budget. ``load_config`` resolves the
caps into ``config.caps``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

# Visual tokens per frame of budget: a call gets num_frames x TOKENS_PER_FRAME tokens.
TOKENS_PER_FRAME = 256
# Settings are written for this budget: 128 frames x 256 tokens = 32,768 visual tokens (32K).
REFERENCE_FRAMES = 128
REFERENCE_VISUAL_TOKENS = REFERENCE_FRAMES * TOKENS_PER_FRAME
# Qwen3-VL encodes 16-pixel patches and merges 2x2 of them into one visual token.
PIXELS_PER_VISUAL_TOKEN = 32 * 32
POLICIES = ("GLOBAL_ONLY", "GLOBAL_HEAVY", "BALANCED", "LOCAL_HEAVY", "LOCAL_ONLY")


@dataclass
class BudgetConfig:
    """Visual-token budget per MLLM call: ``num_frames`` x 256 tokens (128 = 32K, 64 = 16K)."""

    num_frames: int = REFERENCE_FRAMES

    @property
    def visual_tokens(self) -> int:
        return self.num_frames * TOKENS_PER_FRAME

    @property
    def total_pixels(self) -> int:
        return self.visual_tokens * PIXELS_PER_VISUAL_TOKEN

    @property
    def scale(self) -> Fraction:
        """This budget relative to the 128-frame reference the settings are written for."""
        return Fraction(self.num_frames, REFERENCE_FRAMES)


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"
    dtype: str = "float16"
    # Decoding is greedy for every call.
    max_new_tokens: int = 768
    answer_parse_attempts: int = 3
    plan_parse_attempts: int = 2


@dataclass
class VideoConfig:
    """Per-frame pixel bounds shared by every Qwen call."""

    min_frames: int = 4
    min_pixels: int = 131072
    max_pixels: int = 524288
    # Raw frames are budgeted at this multiple of the token budget's pixels (two frames share a
    # temporal patch).
    raw_pixels_multiplier: float = 2.0


@dataclass
class FramesConfig:
    """Frame caps as multiples of ``budget.num_frames`` (1 = num_frames frames)."""

    planner: float = 1  # sparse whole-video overview the planner sees
    refiner: float = 2  # frames sampled from candidate regions for the refiner
    local: float = 1  # final answer, local (detail) stream
    overview: float = 2  # final answer, overview stream of GLOBAL_ONLY, GLOBAL_HEAVY and BALANCED
    overview_local_heavy: float = 1  # final answer, overview stream of LOCAL_HEAVY


@dataclass
class FrameCaps:
    """Frame caps in frames, resolved from ``frames`` and ``budget.num_frames`` (not set in configs)."""

    planner: int
    refiner: int
    local: int
    full_coverage: int
    overview: dict[str, int]


@dataclass
class FpsConfig:
    overview: float = 1.0  # planner overview and final overview stream
    local: float = 2.0  # final local stream
    refiner: float = 1.0  # candidate-region frames for the refiner


@dataclass
class TokensPerFrameConfig:
    """Minimum visual tokens per frame (fixed across budgets, so frames keep their resolution)."""

    # Final overview stream per policy. A policy without an entry (LOCAL_HEAVY) samples its overview at
    # the overview fps up to its frame cap with no floor. LOCAL_ONLY has no overview stream unless it
    # finds no local evidence, in which case it answers from an overview exactly as GLOBAL_ONLY does.
    overview: dict[str, int] = field(default_factory=lambda: {"GLOBAL_ONLY": 64, "GLOBAL_HEAVY": 64, "BALANCED": 60})
    local: dict[str, int] = field(default_factory=lambda: {"LOCAL_HEAVY": 128})


@dataclass
class FullCoverageConfig:
    frames: float = 4
    fps: float = 2.0
    max_stride_sec: float = 2.0
    min_tokens_per_frame: int = 64
    tokens_per_frame_ladder: list[int] = field(default_factory=lambda: [160, 128, 112, 96, 88, 80, 64])


@dataclass
class RetrievalConfig:
    universal_vtg_root: str | None = None  # defaults to $ADAALLOC_UNIVERSAL_VTG_ROOT or third_party/universalvtg
    experiment: str = "experiments/universalvtg"  # relative to universal_vtg_root
    checkpoint: str = "best"
    device: str = "cuda"
    feature_fps: float = 2.0
    top_k: int = 5


@dataclass
class MemoryConfig:
    max_regions: int = 12
    merge_gap_sec: float = 1.0
    padding_sec: float = 0.5
    min_region_confidence: float = 0.2
    max_segments_per_query: int = 5
    max_question_segments: int = 5
    max_planner_seeds: int = 3
    # Answer from uniform frames when every retrieved moment scores below this.
    weak_retrieval_threshold: float = 0.35


@dataclass
class RefinerConfig:
    max_new_tokens: int = 1024
    parse_attempts: int = 2
    min_frames_per_region: int = 8
    keep_score_strength: float = 0.8
    min_keep_confidence: float = 0.0
    min_adjust_confidence: float = 0.85
    min_remove_confidence: float = 0.8
    max_uncertainty_for_removal: float = 0.5
    min_remove_frames: int = 3
    min_merge_confidence: float = 0.8
    max_merge_gap_sec: float = 15.0
    max_focus_regions: int = 2
    block_removes_for_order_count: bool = True
    snap_boundary_frames: bool = True
    # Answer from uniform frames when the refiner finds no usable evidence.
    no_evidence_fallback: bool = True


@dataclass
class DatasetConfig:
    name: str
    annotations: str
    video_ids: str
    video_paths: str
    video_info: str | None = None
    pecore_features: str | None = None


@dataclass
class Config:
    name: str = "adaalloc"
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    # False: every question uses ``fixed_policy`` and retrieval always runs (BALANCED is the paper's
    # Fixed-policy ablation, LOCAL_ONLY its Local-only one).
    adaptive_allocation: bool = True
    fixed_policy: str = "BALANCED"
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    frames: FramesConfig = field(default_factory=FramesConfig)
    fps: FpsConfig = field(default_factory=FpsConfig)
    tokens_per_frame: TokensPerFrameConfig = field(default_factory=TokensPerFrameConfig)
    full_coverage: FullCoverageConfig = field(default_factory=FullCoverageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    dataset: DatasetConfig | None = None
    caps: FrameCaps | None = None  # resolved by load_config


def _frames(multiple: float, num_frames: int, name: str) -> int:
    if isinstance(multiple, bool) or not isinstance(multiple, (int, float)):
        raise ValueError(f"{name} must be a number (a multiple of budget.num_frames), got {multiple!r}")
    frames = Fraction(str(multiple)) * num_frames
    if frames.denominator != 1 or frames < 1:
        raise ValueError(f"{name}={multiple} x budget.num_frames={num_frames} is not a whole, positive number of frames")
    return int(frames)


def apply_budget(config: Config) -> Config:
    """Resolve the frame caps for ``config.budget.num_frames``."""

    num_frames = config.budget.num_frames
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError(f"budget.num_frames must be a positive whole number of frames, got {num_frames!r}")
    if not isinstance(config.adaptive_allocation, bool):
        raise ValueError(f"adaptive_allocation must be true or false, got {config.adaptive_allocation!r}")
    if config.fixed_policy not in POLICIES:
        raise ValueError(f"fixed_policy must be one of {POLICIES}, got {config.fixed_policy!r}")
    frames = config.frames
    overview = _frames(frames.overview, num_frames, "frames.overview")
    config.caps = FrameCaps(
        planner=_frames(frames.planner, num_frames, "frames.planner"),
        refiner=_frames(frames.refiner, num_frames, "frames.refiner"),
        local=_frames(frames.local, num_frames, "frames.local"),
        full_coverage=_frames(config.full_coverage.frames, num_frames, "full_coverage.frames"),
        overview={
            "GLOBAL_ONLY": overview,
            "GLOBAL_HEAVY": overview,
            "BALANCED": overview,
            "LOCAL_HEAVY": _frames(frames.overview_local_heavy, num_frames, "frames.overview_local_heavy"),
        },
    )
    config.full_coverage.max_stride_sec = float(config.full_coverage.max_stride_sec / config.budget.scale)
    return config


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a config, following ``base:`` (relative to the file) and merging this file over it."""

    raw = yaml.safe_load(path.read_text()) or {}
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _merge(_read_yaml((path.parent / base).resolve()), raw)


def _build(cls, values: dict[str, Any]):
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else known[name].default
        kwargs[name] = _build(type(default), value) if is_dataclass(default) and isinstance(value, dict) else value
    return cls(**kwargs)


def load_dataset(path: str | Path, data_root: str | Path | None = None) -> DatasetConfig:
    """Load a dataset config; relative paths resolve against ``data_root`` (default: the file's directory)."""

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    root = Path(data_root).resolve() if data_root is not None else path.parent
    for key in ("annotations", "video_ids", "video_paths", "video_info", "pecore_features"):
        if raw.get(key) is not None and not Path(raw[key]).is_absolute():
            raw[key] = str(root / raw[key])
    return DatasetConfig(**raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a method config (with ``base:`` inheritance and dotted overrides) and apply the budget."""

    raw = _read_yaml(Path(path).resolve())
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    if "caps" in raw:
        raise ValueError("caps are derived from frames and budget.num_frames; set those instead")
    return apply_budget(_build(Config, raw))
