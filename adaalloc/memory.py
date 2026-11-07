"""Temporal evidence memory: build the candidate-region bank, apply the refiner's actions, split the budget."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Sequence

from .config import Config
from .data import QuestionSample, RetrievedSegment
from .planning import AdaptiveMemoryRefinement, AdaptiveOverviewPlan, AdaptiveRefinementAction
from .retrieval import sample_frames_from_retrieved_segments

QUESTION_TIME_REF = "question_time_ref"
OVERVIEW_PROPOSAL = "overview_proposal"
UVTG_SEGMENT = "uvtg_segment"
UVTG_ORIGINAL_QUESTION_SEGMENT = "uvtg_original_question_segment"
_SOURCE_PRIORITY = {
    QUESTION_TIME_REF: 0,
    OVERVIEW_PROPOSAL: 1,
    UVTG_ORIGINAL_QUESTION_SEGMENT: 2,
    UVTG_SEGMENT: 3,
}


@dataclass(slots=True)
class MemoryBankRegion:
    region_id: str
    start_sec: float
    end_sec: float
    sources: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    frame_indices: list[int] = field(default_factory=list)
    query_type: str = ""
    candidate_rank: int | None = None
    candidate_details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryBank:
    regions: list[MemoryBankRegion]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"regions": [region.to_dict() for region in self.regions], **self.metadata}


@dataclass(slots=True)
class AdaptiveAllocation:
    overview_ratio: float
    local_ratio: float
    overview_pixels: int
    local_pixels: int
    metadata: dict[str, Any]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _source_priority(sources: Sequence[str]) -> int:
    if not sources:
        return 99
    return min(_SOURCE_PRIORITY.get(source, 99) for source in sources)


def _parse_timestamp(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return None


def _looks_like_non_video_clock_reference(value: str) -> bool:
    normalized = value.casefold()
    video_time_markers = [
        "in the video",
        "into the video",
        "video timestamp",
        "video time",
        "video mark",
        "clip timestamp",
        "clip time",
    ]
    if any(marker in normalized for marker in video_time_markers):
        return False
    non_video_clock_markers = [
        "quarter",
        "game clock",
        "match time",
        "scoreboard",
        "football match",
        "basketball",
        "soccer",
        "player number",
        "jersey number",
    ]
    return any(marker in normalized for marker in non_video_clock_markers)


def parse_time_reference(value: str | None) -> tuple[list[tuple[float, float]], str | None]:
    """Original-video time spans named in ``value``, and the text itself when none could be parsed."""
    if not value:
        return [], None
    spans: list[tuple[float, float]] = []
    timestamp_pattern = r"\d+(?::\d+){0,2}(?:\.\d+)?"
    colon_timestamp_pattern = r"\d+:\d+(?::\d+)?(?:\.\d+)?"
    seconds_timestamp_pattern = r"\d+(?:\.\d+)?"
    range_patterns = [
        rf"({timestamp_pattern})\s*(?:seconds?|secs?|s)?\s*[-–—]\s*({timestamp_pattern})\s*(?:seconds?|secs?|s)?",
        rf"(?:from|between)\s+({timestamp_pattern})\s*(?:seconds?|secs?|s)?\s+(?:to|and)\s+({timestamp_pattern})\s*(?:seconds?|secs?|s)?",
    ]
    for pattern in range_patterns:
        for start_text, end_text in re.findall(pattern, value, flags=re.IGNORECASE):
            start = _parse_timestamp(start_text)
            end = _parse_timestamp(end_text)
            if start is not None and end is not None and end > start:
                spans.append((start, end))
    if spans:
        deduped = list(dict.fromkeys(spans))
        return deduped, None
    for start_text, end_text in re.findall(rf"({timestamp_pattern})\s*[-–—]\s*({timestamp_pattern})", value):
        start = _parse_timestamp(start_text)
        end = _parse_timestamp(end_text)
        if start is not None and end is not None and end > start:
            spans.append((start, end))
    if spans:
        return spans, None
    if _looks_like_non_video_clock_reference(value):
        return [], value
    single_patterns = [
        rf"(?:at|around|near|by|timestamp|time|mark)\s+(?:the\s+)?(?:timestamp\s+|time\s+|mark\s+)?({colon_timestamp_pattern})\b",
        rf"({colon_timestamp_pattern})\s*(?:timestamp|time|mark)\b",
        rf"(?:at|around|near|by|timestamp|time|mark)\s+(?:the\s+)?(?:timestamp\s+|time\s+|mark\s+)?({seconds_timestamp_pattern})\s*(?:seconds?|secs?|s)\b",
        rf"({seconds_timestamp_pattern})\s*(?:seconds?|secs?|s)\s+(?:into|in)\s+(?:the\s+)?(?:video|clip|recording)\b",
    ]
    for pattern in single_patterns:
        for match in re.finditer(pattern, value, flags=re.IGNORECASE):
            timestamp_text = match.group(1)
            timestamp = _parse_timestamp(timestamp_text)
            if timestamp is not None:
                spans.append((timestamp, timestamp + 1.0))
    if spans:
        deduped = list(dict.fromkeys(spans))
        return deduped, None
    single = _parse_timestamp(value)
    if single is not None:
        return [(single, single + 1.0)], None
    return [], value


def _duration(sample: QuestionSample) -> float | None:
    return sample.video_metadata.duration_sec


def normalize_region(region: MemoryBankRegion, *, duration_sec: float | None, padding_sec: float = 0.0) -> MemoryBankRegion | None:
    start_sec = region.start_sec - padding_sec
    end_sec = region.end_sec + padding_sec
    if duration_sec is not None:
        start_sec = _clamp(start_sec, 0.0, duration_sec)
        end_sec = _clamp(end_sec, 0.0, duration_sec)
    else:
        start_sec = max(0.0, start_sec)
        end_sec = max(0.0, end_sec)
    if end_sec <= start_sec:
        return None
    return MemoryBankRegion(
        region_id=region.region_id,
        start_sec=start_sec,
        end_sec=end_sec,
        sources=list(dict.fromkeys(region.sources)),
        queries=list(dict.fromkeys(query for query in region.queries if query)),
        score=float(region.score),
        confidence=float(region.confidence),
        rationale=region.rationale,
        frame_indices=sorted(set(int(index) for index in region.frame_indices)),
        query_type=region.query_type,
        candidate_rank=region.candidate_rank,
        candidate_details=[dict(detail) for detail in region.candidate_details],
    )


def _merge_regions(left: MemoryBankRegion, right: MemoryBankRegion) -> MemoryBankRegion:
    return MemoryBankRegion(
        region_id=left.region_id,
        start_sec=min(left.start_sec, right.start_sec),
        end_sec=max(left.end_sec, right.end_sec),
        sources=list(dict.fromkeys([*left.sources, *right.sources])),
        queries=list(dict.fromkeys([*left.queries, *right.queries])),
        score=max(float(left.score), float(right.score)),
        confidence=max(float(left.confidence), float(right.confidence)),
        rationale="; ".join(part for part in [left.rationale, right.rationale] if part),
        frame_indices=sorted(set([*left.frame_indices, *right.frame_indices])),
        query_type=left.query_type if left.query_type == right.query_type else "mixed",
        candidate_rank=min(
            (rank for rank in [left.candidate_rank, right.candidate_rank] if rank is not None),
            default=None,
        ),
        candidate_details=[*left.candidate_details, *right.candidate_details],
    )


def _segment_memory_source(segment: RetrievedSegment) -> str:
    if segment.source == "universal_vtg_original_question":
        return UVTG_ORIGINAL_QUESTION_SEGMENT
    return UVTG_SEGMENT


def _segment_query_type(segment: RetrievedSegment) -> str:
    if _segment_memory_source(segment) == UVTG_ORIGINAL_QUESTION_SEGMENT:
        return "original_question"
    return "generated"


def _segment_candidate_detail(segment: RetrievedSegment, *, rank: int) -> dict[str, Any]:
    return {
        "query": segment.query,
        "query_type": _segment_query_type(segment),
        "rank": int(rank),
        "source": _segment_memory_source(segment),
        "score": float(segment.score),
        "start_sec": float(segment.start_sec),
        "end_sec": float(segment.end_sec),
    }


def _sort_regions_by_score(regions: Iterable[MemoryBankRegion]) -> list[MemoryBankRegion]:
    return sorted(
        regions,
        key=lambda region: (
            -float(region.score),
            -float(region.confidence),
            region.start_sec,
            region.end_sec,
            region.region_id,
        ),
    )


def _source_counter(regions: Iterable[MemoryBankRegion]) -> dict[str, int]:
    counts = {QUESTION_TIME_REF: 0, OVERVIEW_PROPOSAL: 0, UVTG_SEGMENT: 0, UVTG_ORIGINAL_QUESTION_SEGMENT: 0}
    for region in regions:
        for source in region.sources:
            counts[source] = counts.get(source, 0) + 1
    return counts


def _apply_source_aware_region_limits(
    regions: list[MemoryBankRegion],
    *,
    max_segments_per_generated_query: int,
    max_original_question_segments: int,
    max_overview_proposals: int,
) -> tuple[list[MemoryBankRegion], dict[str, Any]]:
    question_regions: list[MemoryBankRegion] = []
    overview_regions: list[MemoryBankRegion] = []
    generated_segment_regions: list[MemoryBankRegion] = []
    original_question_regions: list[MemoryBankRegion] = []
    other_regions: list[MemoryBankRegion] = []

    for region in regions:
        primary_source = region.sources[0] if region.sources else ""
        if primary_source == QUESTION_TIME_REF:
            question_regions.append(region)
        elif primary_source == OVERVIEW_PROPOSAL:
            overview_regions.append(region)
        elif primary_source == UVTG_ORIGINAL_QUESTION_SEGMENT:
            original_question_regions.append(region)
        elif primary_source == UVTG_SEGMENT:
            generated_segment_regions.append(region)
        else:
            other_regions.append(region)

    kept_overview = _sort_regions_by_score(overview_regions)[:max_overview_proposals]
    kept_original = _sort_regions_by_score(original_question_regions)[:max_original_question_segments]

    generated_by_query: dict[str, list[MemoryBankRegion]] = {}
    for region in generated_segment_regions:
        key = region.queries[0] if region.queries else ""
        generated_by_query.setdefault(key, []).append(region)
    kept_generated: list[MemoryBankRegion] = []
    for query in sorted(generated_by_query):
        kept_generated.extend(_sort_regions_by_score(generated_by_query[query])[:max_segments_per_generated_query])

    kept = [*question_regions, *kept_overview, *kept_original, *kept_generated, *other_regions]
    input_counts = _source_counter(regions)
    kept_counts = _source_counter(kept)
    dropped_counts = {source: max(0, input_counts.get(source, 0) - kept_counts.get(source, 0)) for source in sorted(input_counts)}
    return kept, {
        "enabled": True,
        "limits": {
            "max_segments_per_generated_query": int(max_segments_per_generated_query),
            "max_original_question_segments": int(max_original_question_segments),
            "max_overview_proposals": int(max_overview_proposals),
        },
        "input_counts": input_counts,
        "kept_counts": kept_counts,
        "dropped_counts": dropped_counts,
    }


def normalize_merge_and_cap_regions(
    regions: Iterable[MemoryBankRegion],
    *,
    duration_sec: float | None,
    padding_sec: float,
    merge_gap_sec: float,
    max_regions: int,
) -> tuple[list[MemoryBankRegion], dict[str, Any]]:
    """Pad and clamp regions, merge those closer than ``merge_gap_sec``, keep the ``max_regions`` best, renumber r1.."""
    normalized = [region for region in (normalize_region(region, duration_sec=duration_sec, padding_sec=padding_sec) for region in regions) if region is not None]
    normalized.sort(key=lambda region: (region.start_sec, region.end_sec, _source_priority(region.sources), region.region_id))
    merged: list[MemoryBankRegion] = []
    for region in normalized:
        if merged and region.start_sec - merged[-1].end_sec <= merge_gap_sec:
            merged[-1] = _merge_regions(merged[-1], region)
        else:
            merged.append(region)
    capped = sorted(
        merged,
        key=lambda region: (
            _source_priority(region.sources),
            -float(region.score),
            -float(region.confidence),
            region.start_sec,
            region.region_id,
        ),
    )[:max_regions]
    capped.sort(key=lambda region: (region.start_sec, region.end_sec, region.region_id))
    for idx, region in enumerate(capped, start=1):
        region.region_id = f"r{idx}"
    return capped, {"raw_region_count": len(normalized), "merged_region_count": len(merged), "capped_region_count": len(capped)}


def build_memory_bank(
    sample: QuestionSample,
    plan: AdaptiveOverviewPlan,
    retrieved_segments: list[RetrievedSegment],
    config: Config,
) -> MemoryBank:
    """Candidate regions from the planner's proposals and the retrieved segments, limited per source and merged."""
    regions: list[MemoryBankRegion] = []
    for index, proposal in enumerate(plan.temporal_proposals, start=1):
        proposal_source = QUESTION_TIME_REF if proposal.source == "question" else OVERVIEW_PROPOSAL
        regions.append(
            MemoryBankRegion(
                region_id=f"p{index}",
                start_sec=proposal.start_sec,
                end_sec=proposal.end_sec,
                sources=[proposal_source],
                queries=list(plan.queries) or [sample.question_text],
                score=proposal.confidence,
                confidence=proposal.confidence,
                rationale=proposal.rationale or proposal.label,
            )
        )
    candidate_rank_by_query: dict[str, int] = {}
    for index, segment in enumerate(retrieved_segments, start=1):
        segment_source = _segment_memory_source(segment)
        query_key = f"{_segment_query_type(segment)}::{segment.query}"
        candidate_rank_by_query[query_key] = candidate_rank_by_query.get(query_key, 0) + 1
        candidate_rank = candidate_rank_by_query[query_key]
        regions.append(
            MemoryBankRegion(
                region_id=f"u{index}",
                start_sec=float(segment.start_sec),
                end_sec=float(segment.end_sec),
                sources=[segment_source],
                queries=[segment.query],
                score=float(segment.score),
                confidence=float(segment.score),
                rationale=(
                    f"UniversalVTG original-question fallback segment rank {candidate_rank}."
                    if segment_source == UVTG_ORIGINAL_QUESTION_SEGMENT
                    else f"UniversalVTG retrieved segment rank {candidate_rank}."
                ),
                frame_indices=list(segment.frame_indices),
                query_type=_segment_query_type(segment),
                candidate_rank=candidate_rank,
                candidate_details=[_segment_candidate_detail(segment, rank=candidate_rank)],
            )
        )
    memory = config.memory
    regions, source_aware_metadata = _apply_source_aware_region_limits(
        regions,
        max_segments_per_generated_query=int(memory.max_segments_per_query),
        max_original_question_segments=int(memory.max_question_segments),
        max_overview_proposals=int(memory.max_planner_seeds),
    )
    normalized, metadata = normalize_merge_and_cap_regions(
        regions,
        duration_sec=_duration(sample),
        padding_sec=float(memory.padding_sec),
        merge_gap_sec=float(memory.merge_gap_sec),
        max_regions=int(memory.max_regions),
    )
    metadata.update(
        {
            "question_time_reference": None,
            "dataset_time_reference_ignored": sample.time_reference is not None,
            "unparsed_question_time_reference": None,
            "source_counts": {
                QUESTION_TIME_REF: 0,
                OVERVIEW_PROPOSAL: sum(1 for proposal in plan.temporal_proposals if proposal.source != "question"),
                "question_proposal": sum(1 for proposal in plan.temporal_proposals if proposal.source == "question"),
                UVTG_SEGMENT: sum(1 for segment in retrieved_segments if _segment_memory_source(segment) == UVTG_SEGMENT),
                UVTG_ORIGINAL_QUESTION_SEGMENT: sum(1 for segment in retrieved_segments if _segment_memory_source(segment) == UVTG_ORIGINAL_QUESTION_SEGMENT),
            },
            "kept_source_counts": _source_counter(regions),
            "source_aware_memory_ranking": source_aware_metadata,
        }
    )
    return MemoryBank(regions=normalized, metadata=metadata)


def regions_to_retrieved_segments(regions: Iterable[MemoryBankRegion]) -> list[RetrievedSegment]:
    segments: list[RetrievedSegment] = []
    for region in regions:
        query = region.queries[0] if region.queries else region.rationale or region.region_id
        segments.append(
            RetrievedSegment(
                query=query,
                start_sec=region.start_sec,
                end_sec=region.end_sec,
                score=max(float(region.score), float(region.confidence)),
                frame_indices=list(region.frame_indices),
                source="adaptive_memory:" + "+".join(region.sources),
            )
        )
    return segments


def candidate_frame_indices_by_region(
    regions: Iterable[MemoryBankRegion],
    frames: Iterable[Any],
    *,
    tolerance_sec: float = 0.0,
) -> dict[str, list[int]]:
    """Frame indices of the candidate frames inside each region's span, widened by `tolerance_sec`."""

    frames = list(frames)
    return {
        region.region_id: [
            int(frame.frame_index)
            for frame in frames
            if float(region.start_sec) - tolerance_sec - 1e-9
            <= float(frame.timestamp_sec)
            <= float(region.end_sec) + tolerance_sec + 1e-9
        ]
        for region in regions
    }


def effective_candidate_max_frames(config: Config) -> int:
    """Frame cap of the refiner's candidate-region clips."""
    return int(config.caps.refiner)


def sample_candidate_frame_indices(
    regions: list[MemoryBankRegion],
    *,
    native_fps: float,
    total_frames: int,
    config: Config,
) -> list[int]:
    """Frames for the refiner: sampled at the refiner fps and shared across regions by length."""
    return sample_frames_from_retrieved_segments(
        regions_to_retrieved_segments(regions),
        native_fps=native_fps,
        sample_fps=float(config.fps.refiner),
        total_frames=total_frames,
        max_frames=effective_candidate_max_frames(config),
        sampling_mode="region_length",
        min_frames_per_region=int(config.refiner.min_frames_per_region),
    )


_ORDER_OR_COUNT_QUESTION = re.compile(
    r"\b(first|last|earliest|before|after|order|sequence|then|how many|count|number of|times)\b",
    re.IGNORECASE,
)


def is_order_or_count_question(question_text: str) -> bool:
    """Order, first/last, before/after, or counting questions, where removing one occurrence can change the answer."""
    return bool(_ORDER_OR_COUNT_QUESTION.search(question_text or ""))


def _apply_guarded_merges(
    actions: list[AdaptiveRefinementAction],
    *,
    region_map: dict[str, MemoryBankRegion],
    replacements: dict[str, MemoryBankRegion],
    removed: set[str],
    merge_members: set[str],
    applied_merge_groups: list[list[str]],
    suppressed_merge_actions: list[dict[str, Any]],
    unknown_ids: list[str],
    refinement_uncertainty: float,
    max_uncertainty: float | None,
    min_confidence: float,
    max_gap_sec: float | None,
) -> None:
    """Union the accepted merge requests into groups and merge each group.

    A request is dropped when its confidence is low, the refiner's overall
    uncertainty exceeds the removal gate, or a member is unknown or removed.
    A group is merged only if consecutive members are at most `max_gap_sec`
    apart; the merged span is the members' combined span.
    """

    parent: dict[str, str] = {}

    def find(region_id: str) -> str:
        while parent.setdefault(region_id, region_id) != region_id:
            parent[region_id] = parent[parent[region_id]]
            region_id = parent[region_id]
        return region_id

    accepted: list[AdaptiveRefinementAction] = []
    for action in actions:
        reason = None
        missing = [region_id for region_id in action.region_ids if region_id not in region_map]
        if missing:
            unknown_ids.extend(missing)
            reason = "unknown_member"
        elif any(region_id in removed for region_id in action.region_ids):
            reason = "member_removed"
        elif float(action.confidence or 0.0) < min_confidence:
            reason = "merge_confidence_below_threshold"
        elif max_uncertainty is not None and refinement_uncertainty > float(max_uncertainty):
            reason = "refinement_uncertainty_above_threshold"
        if reason is not None:
            suppressed = action.to_dict()
            suppressed["suppression_reason"] = reason
            suppressed_merge_actions.append(suppressed)
            continue
        accepted.append(action)
        first, *others = action.region_ids
        for other in others:
            parent[find(other)] = find(first)

    groups: dict[str, list[str]] = {}
    for region_id in parent:
        groups.setdefault(find(region_id), []).append(region_id)
    for members in groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(members, key=lambda region_id: (region_map[region_id].start_sec, region_map[region_id].end_sec, region_id))
        reach = float(region_map[ordered[0]].end_sec)
        too_far = False
        for region_id in ordered[1:]:
            if max_gap_sec is not None and float(region_map[region_id].start_sec) - reach > float(max_gap_sec):
                too_far = True
                break
            reach = max(reach, float(region_map[region_id].end_sec))
        if too_far:
            for action in accepted:
                if set(action.region_ids) & set(members):
                    suppressed = action.to_dict()
                    suppressed["suppression_reason"] = "merge_gap_too_large"
                    suppressed_merge_actions.append(suppressed)
            continue
        merged = replacements.get(ordered[0], region_map[ordered[0]])
        for region_id in ordered[1:]:
            merged = _merge_regions(merged, replacements.get(region_id, region_map[region_id]))
        merged.region_id = min(members)
        for region_id in members:
            replacements.pop(region_id, None)
        replacements[merged.region_id] = merged
        merge_members.update(members)
        applied_merge_groups.append(sorted(members))


def apply_refinement(
    regions: list[MemoryBankRegion],
    refinement: AdaptiveMemoryRefinement,
    *,
    duration_sec: float | None,
    min_remove_confidence: float = 0.0,
    max_uncertainty_for_removal: float | None = None,
    keep_score_strength: float = 1.0,
    min_keep_confidence: float = 0.0,
    min_adjust_confidence: float = 0.0,
    max_focus_regions: int | None = None,
    frame_counts_by_region: dict[str, int] | None = None,
    min_remove_frames: int = 0,
    min_merge_confidence: float = 0.0,
    max_merge_gap_sec: float | None = None,
    block_removes: bool = False,
) -> tuple[list[MemoryBankRegion], dict[str, Any]]:
    """Apply the refiner's actions to the memory bank, each behind its guard.

    - keep: the region's score becomes a blend of its prior score and the refiner's relevance
      (``keep_score_strength`` weights the relevance).
    - adjust: adds a tighter focus region strictly inside its region; the region stays as context.
      At most ``max_focus_regions`` focuses survive, the most confident first.
    - merge: needs ``min_merge_confidence`` and members no more than ``max_merge_gap_sec`` apart, and
      takes the members' combined span rather than model-written boundaries.
    - remove: must pass the refiner's own checks (``query_match`` and ``helpful`` both "no"), the
      confidence and uncertainty gates, and a minimum number of observed frames, and never touches a
      question-time region. ``block_removes`` suppresses every remove, for questions where each
      occurrence can matter (order, first/last, counting).

    Question-time regions and, unless helpful evidence survives, the best original-question segment
    are restored if refinement dropped them all. Returns the renumbered regions and the decision log.
    """

    if not 0 <= float(keep_score_strength) <= 1:
        raise ValueError(
            "keep_score_strength must be between 0.0 and 1.0."
        )
    if (
        max_uncertainty_for_removal is not None
        and not 0 <= float(max_uncertainty_for_removal) <= 1
    ):
        raise ValueError(
            "max_uncertainty_for_removal must be null or between "
            "0.0 and 1.0."
        )
    if not 0 <= float(min_keep_confidence) <= 1:
        raise ValueError(
            "min_keep_confidence must be between 0.0 and 1.0."
        )
    if not 0 <= float(min_adjust_confidence) <= 1:
        raise ValueError(
            "min_adjust_confidence must be between 0.0 and 1.0."
        )
    if max_focus_regions is not None and int(max_focus_regions) <= 0:
        raise ValueError(
            "max_focus_regions must be positive when provided."
        )
    frame_counts = dict(frame_counts_by_region or {})
    guarded_merge_actions: list[AdaptiveRefinementAction] = []
    applied_merge_groups: list[list[str]] = []
    uncertainty_removal_gate_active = bool(
        max_uncertainty_for_removal is not None
        and float(refinement.uncertainty)
        > float(max_uncertainty_for_removal)
    )
    region_map = {region.region_id: region for region in regions}
    removed: set[str] = set()
    replacements: dict[str, MemoryBankRegion] = {}
    merge_members: set[str] = set()
    unknown_ids: list[str] = []
    invalid_actions: list[dict[str, Any]] = []
    suppressed_remove_actions: list[dict[str, Any]] = []
    suppressed_merge_actions: list[dict[str, Any]] = []
    suppressed_adjust_actions: list[dict[str, Any]] = []
    suppressed_keep_score_updates: list[dict[str, Any]] = []
    keep_score_updates: list[dict[str, Any]] = []
    focused_adjustments: dict[str, MemoryBankRegion] = {}
    focus_confidences: dict[str, float] = {}

    def _normalize_action_region(action: AdaptiveRefinementAction, base: MemoryBankRegion) -> MemoryBankRegion | None:
        if action.start_sec is None or action.end_sec is None:
            return None
        region = MemoryBankRegion(
            region_id=base.region_id,
            start_sec=float(action.start_sec),
            end_sec=float(action.end_sec),
            sources=list(base.sources),
            queries=list(base.queries),
            score=base.score,
            confidence=action.confidence if action.confidence is not None else base.confidence,
            rationale=action.rationale or base.rationale,
            frame_indices=list(base.frame_indices),
            query_type=base.query_type,
            candidate_rank=base.candidate_rank,
            candidate_details=[dict(detail) for detail in base.candidate_details],
        )
        return normalize_region(region, duration_sec=duration_sec, padding_sec=0.0)

    for action in refinement.actions:
        if action.action == "keep":
            keep_signal = (
                action.relevance
                if action.relevance is not None
                else action.confidence
            )
            if action.region_id and action.region_id not in region_map:
                unknown_ids.append(action.region_id)
            elif (
                action.region_id in region_map
                and keep_signal is not None
            ):
                base = region_map[action.region_id]
                updated = normalize_region(
                    base,
                    duration_sec=duration_sec,
                    padding_sec=0.0,
                )
                if updated is None:
                    invalid_actions.append(action.to_dict())
                    continue
                relevance = _clamp(float(keep_signal), 0.0, 1.0)
                if relevance < float(min_keep_confidence):
                    suppressed_action = action.to_dict()
                    suppressed_action["suppression_reason"] = (
                        "keep_confidence_below_threshold"
                    )
                    suppressed_keep_score_updates.append(
                        suppressed_action
                    )
                    continue
                prior_score = _clamp(
                    max(float(base.score), float(base.confidence)),
                    0.0,
                    1.0,
                )
                updated_score = (
                    (1.0 - float(keep_score_strength)) * prior_score
                    + float(keep_score_strength) * relevance
                )
                updated_score = _clamp(updated_score, 0.0, 1.0)
                updated.score = updated_score
                updated.confidence = updated_score
                if action.rationale:
                    updated.rationale = action.rationale
                replacements[action.region_id] = updated
                keep_score_updates.append(
                    {
                        "region_id": action.region_id,
                        "previous_score": float(base.score),
                        "previous_confidence": float(base.confidence),
                        "refined_relevance": relevance,
                        "update_strength": float(
                            keep_score_strength
                        ),
                        "updated_score": updated_score,
                    }
                )
            continue
        if action.action == "remove":
            action_confidence = float(action.confidence or 0.0)
            suppression_reason = None
            if action.region_id not in region_map:
                # Usually a context clip outside every region; nothing to remove.
                suppression_reason = "not_a_memory_region"
            elif uncertainty_removal_gate_active:
                suppression_reason = "refinement_uncertainty_above_threshold"
            elif action_confidence < float(min_remove_confidence):
                suppression_reason = "remove_confidence_below_threshold"
            elif not (
                action.query_match == "no" and action.helpful == "no"
            ):
                suppression_reason = "remove_checks_not_both_no"
            elif int(
                frame_counts.get(str(action.region_id), 0)
            ) < int(min_remove_frames):
                suppression_reason = "too_few_observed_frames"
            elif QUESTION_TIME_REF in region_map[action.region_id].sources:
                suppression_reason = "question_time_region_protected"
            elif block_removes:
                suppression_reason = "order_or_count_question"
            if suppression_reason is not None:
                suppressed_action = action.to_dict()
                suppressed_action["suppression_reason"] = suppression_reason
                suppressed_remove_actions.append(suppressed_action)
                continue
            removed.add(action.region_id)  # type: ignore[arg-type]
            continue
        if action.action == "adjust":
            if action.region_id not in region_map:
                if action.region_id:
                    unknown_ids.append(action.region_id)
                continue
            action_confidence = float(action.confidence or 0.0)
            if action_confidence < float(min_adjust_confidence):
                suppressed_action = action.to_dict()
                suppressed_action["suppression_reason"] = (
                    "adjust_confidence_below_threshold"
                )
                suppressed_adjust_actions.append(suppressed_action)
                continue
            base_region = region_map[action.region_id]
            adjusted = _normalize_action_region(action, base_region)
            if adjusted is None:
                invalid_actions.append(action.to_dict())
            elif not (
                float(adjusted.start_sec)
                >= float(base_region.start_sec) - 1e-6
                and float(adjusted.end_sec)
                <= float(base_region.end_sec) + 1e-6
                and (
                    float(adjusted.end_sec) - float(adjusted.start_sec)
                    < float(base_region.end_sec)
                    - float(base_region.start_sec)
                    - 1e-6
                )
            ):
                suppressed_action = action.to_dict()
                suppressed_action["suppression_reason"] = (
                    "focus_not_tighter_inside_source_region"
                )
                suppressed_adjust_actions.append(suppressed_action)
            else:
                adjusted.region_id = f"{action.region_id}__focus"
                adjusted.sources = list(
                    dict.fromkeys(
                        [
                            *adjusted.sources,
                            "adaptive_refinement_focus",
                        ]
                    )
                )
                focused_adjustments[action.region_id] = adjusted
                focus_confidences[action.region_id] = action_confidence
            continue
        if action.action == "merge":
            guarded_merge_actions.append(action)

    if (
        max_focus_regions is not None
        and len(focused_adjustments) > int(max_focus_regions)
    ):
        ranked_focus_ids = sorted(
            focused_adjustments,
            key=lambda region_id: (
                -float(focus_confidences.get(region_id, 0.0)),
                float(focused_adjustments[region_id].start_sec),
                region_id,
            ),
        )
        for region_id in ranked_focus_ids[int(max_focus_regions) :]:
            suppressed_adjust_actions.append(
                {
                    "action": "adjust",
                    "region_id": region_id,
                    "confidence": float(
                        focus_confidences.get(region_id, 0.0)
                    ),
                    "suppression_reason": "max_focus_regions_reached",
                }
            )
            focused_adjustments.pop(region_id, None)
            focus_confidences.pop(region_id, None)

    if guarded_merge_actions:
        _apply_guarded_merges(
            guarded_merge_actions,
            region_map=region_map,
            replacements=replacements,
            removed=removed,
            merge_members=merge_members,
            applied_merge_groups=applied_merge_groups,
            suppressed_merge_actions=suppressed_merge_actions,
            unknown_ids=unknown_ids,
            refinement_uncertainty=float(refinement.uncertainty),
            max_uncertainty=max_uncertainty_for_removal,
            min_confidence=float(min_merge_confidence),
            max_gap_sec=max_merge_gap_sec,
        )

    output: list[MemoryBankRegion] = []
    for region in regions:
        if region.region_id in removed or region.region_id in merge_members:
            continue
        output.append(replacements.get(region.region_id, region))
    for region_id, replacement in replacements.items():
        if region_id in merge_members:
            output.append(replacement)
    output.extend(focused_adjustments.values())

    restored_region_ids: list[str] = []
    restored_sources: list[str] = []

    def _contains_source(candidates: Iterable[MemoryBankRegion], source: str) -> bool:
        return any(source in candidate.sources for candidate in candidates)

    def _restore_region(region: MemoryBankRegion) -> None:
        if any(existing.region_id == region.region_id for existing in output):
            return
        output.append(region)
        restored_region_ids.append(region.region_id)
        restored_sources.extend(region.sources)

    # Explicit question time spans are user-provided localization constraints.
    # The VLM may tighten/merge them, but it should not be able to erase them
    # entirely; otherwise timestamped questions can lose their only reliable
    # source of temporal evidence.
    if not _contains_source(output, QUESTION_TIME_REF):
        for region in regions:
            if QUESTION_TIME_REF in region.sources:
                _restore_region(region)

    # Preserve at least one original-question UniversalVTG candidate when it
    # existed but refinement removed all of them, unless evidence the refiner
    # judged helpful survives. Original-question retrieval is often the least
    # assumption-heavy temporal cue.
    helpful_region_ids = {
        str(action.region_id)
        for action in refinement.actions
        if action.helpful in {"yes", "partly"} and action.region_id
    }
    for group in applied_merge_groups:
        if helpful_region_ids.intersection(group):
            helpful_region_ids.add(min(group))
    helpful_evidence_survives = any(
        region.region_id.split("__", 1)[0] in helpful_region_ids
        for region in output
    )
    if not _contains_source(output, UVTG_ORIGINAL_QUESTION_SEGMENT) and not helpful_evidence_survives:
        original_question_regions = [region for region in regions if UVTG_ORIGINAL_QUESTION_SEGMENT in region.sources]
        if original_question_regions:
            _restore_region(_sort_regions_by_score(original_question_regions)[0])

    if not output and regions:
        _restore_region(_sort_regions_by_score(regions)[0])

    output.sort(key=lambda region: (region.start_sec, region.end_sec, region.region_id))
    # Renumber copies: kept regions are the caller's objects, which are logged
    # as the pre-refinement memory bank and must keep their original ids.
    output = [replace(region, region_id=f"r{idx}") for idx, region in enumerate(output, start=1)]
    return output, {
        "action_count": len(refinement.actions),
        "unknown_region_ids": sorted(set(unknown_ids)),
        "invalid_actions": invalid_actions,
        "min_remove_confidence": float(min_remove_confidence),
        "max_uncertainty_for_removal": (
            float(max_uncertainty_for_removal)
            if max_uncertainty_for_removal is not None
            else None
        ),
        "refinement_uncertainty": float(refinement.uncertainty),
        "uncertainty_removal_gate_active": (
            uncertainty_removal_gate_active
        ),
        "suppressed_remove_actions": suppressed_remove_actions,
        "suppressed_merge_actions": suppressed_merge_actions,
        "suppressed_adjust_actions": suppressed_adjust_actions,
        "keep_score_strength": float(keep_score_strength),
        "min_keep_confidence": float(min_keep_confidence),
        "suppressed_keep_score_updates": (
            suppressed_keep_score_updates
        ),
        "keep_score_updates": keep_score_updates,
        "min_adjust_confidence": float(min_adjust_confidence),
        "max_focus_regions": (
            int(max_focus_regions)
            if max_focus_regions is not None
            else None
        ),
        "focused_adjustment_region_ids": sorted(
            focused_adjustments
        ),
        "output_region_count": len(output),
        "removed_region_ids": sorted(removed),
        "merged_region_groups": applied_merge_groups,
        "usable_local_evidence": refinement.usable_local_evidence,
        "restored_region_ids": restored_region_ids,
        "restored_sources": sorted(set(restored_sources)),
        "parse_failed": refinement.parse_failed,
        "fallback_reason": refinement.fallback_reason,
    }


def allocate_visual_budget(
    plan: AdaptiveOverviewPlan,
    *,
    uncertainty: float,
    has_local_evidence: bool,
    total_pixel_budget: int,
    fallback_reasons: list[str] | None = None,
) -> AdaptiveAllocation:
    """Split the final pixel budget into overview and local shares by the planner's policy ratio.

    Without local evidence the whole budget goes to the overview.
    """
    fallback_reasons = list(fallback_reasons or [])
    if not has_local_evidence:
        overview_ratio = 1.0
        mode = "overview_only"
    else:
        if plan.budget_policy.overview_ratio is not None:
            overview_ratio = _clamp(float(plan.budget_policy.overview_ratio), 0.0, 1.0)
        else:
            overview_ratio = _clamp(float(plan.budget_policy.overview_min_ratio), 0.0, 1.0)
        mode = "policy_ratio"
    overview_ratio = _clamp(overview_ratio, 0.0, 1.0)
    local_ratio = 0.0 if not has_local_evidence else max(0.0, 1.0 - overview_ratio)
    overview_pixels = int(total_pixel_budget * overview_ratio)
    local_pixels = max(0, int(total_pixel_budget) - overview_pixels)
    if not has_local_evidence:
        local_pixels = 0
        overview_pixels = int(total_pixel_budget)
    return AdaptiveAllocation(
        overview_ratio=overview_ratio,
        local_ratio=local_ratio,
        overview_pixels=overview_pixels,
        local_pixels=local_pixels,
        metadata={
            "mode": mode,
            "overview_ratio": overview_ratio,
            "local_ratio": local_ratio,
            "overview_pixels": overview_pixels,
            "local_pixels": local_pixels,
            "uncertainty": _clamp(float(uncertainty), 0.0, 1.0),
            "plan_overview_min_ratio": float(plan.budget_policy.overview_min_ratio),
            "fallback_reasons": fallback_reasons,
            "has_local_evidence": has_local_evidence,
        },
    )
