"""Planner and refiner output types, and parsing of planner, refiner and answer replies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

from .data import AnswerPrediction, QuestionSample


def extract_json_object(text: str) -> dict[str, Any]:
    """The reply as a JSON object, else the first decodable ``{...}`` inside it."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    start_positions = [m.start() for m in re.finditer(r"\{", text)]
    for start in start_positions:
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def extract_outer_json_object(text: str) -> dict[str, Any]:
    """Parse the reply's outermost JSON object without salvaging nested ones.

    `extract_json_object` falls back to the first `{` that decodes, so a
    truncated reply can come back as one of its inner objects (for example a
    single list item). This variant only accepts the object that starts at the
    reply's first `{`, tolerating markdown fences and trailing text, and raises
    otherwise so callers can retry.
    """

    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output.")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model output is not one complete JSON object ({exc.msg})."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


# Function words and generic video/person nouns that carry no option-specific
# content; an option is "restated" only through its remaining content words.
_STOPWORDS = frozenset(
    "a an the of to in on at for with and or is are was were be been by from as "
    "that this it its into than then there their they he she his her them video "
    "shown showed show shows person man woman someone".split()
)


def _content_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9']+", str(text).lower())
        if word not in _STOPWORDS and len(word) > 1
    }


def option_restated_by_query(
    query: str,
    options: Iterable[str],
    question_text: str,
    *,
    min_overlap: float = 0.6,
) -> int | None:
    """Return the index of the first answer option a retrieval query restates.

    Only option words that do not already appear in the question count, so a
    query that merely repeats the question's own wording is not flagged. The
    query restates an option when it contains at least ``min_overlap`` of those
    option-only words. This catches paraphrased guesses such as "The weather
    was sunny" for the option "Sunny", which verbatim phrase matching misses.
    """

    query_words = _content_words(query)
    question_words = _content_words(question_text)
    for index, option in enumerate(options):
        option_words = _content_words(option) - question_words
        if option_words and len(option_words & query_words) / len(option_words) >= min_overlap:
            return index
    return None


LETTER_PATTERN = re.compile(r"^[A-Z]$")
LETTER_PREFIX_PATTERN = re.compile(r"^\s*([A-Z])[\.\):\-\s]")


def normalize_option_letter(letter: str | None) -> str | None:
    if letter is None:
        return None
    candidate = letter.strip().upper()
    return candidate if LETTER_PATTERN.match(candidate) else None


def option_letter_to_index(letter: str | None) -> int | None:
    normalized = normalize_option_letter(letter)
    if normalized is None:
        return None
    return ord(normalized) - ord("A")


def option_index_to_letter(index: int | None) -> str | None:
    if index is None or index < 0:
        return None
    return chr(ord("A") + index)


def option_text_for_letter(sample: QuestionSample, letter: str | None) -> str | None:
    index = option_letter_to_index(letter)
    if index is None or index >= len(sample.options):
        return None
    return sample.options[index]


def option_letter_for_text(sample: QuestionSample, text: str | None) -> str | None:
    if not text:
        return None
    candidate = text.strip()
    if not candidate:
        return None

    normalized = normalize_option_letter(candidate)
    if normalized is not None and option_letter_to_index(normalized) is not None and option_letter_to_index(normalized) < len(sample.options):
        return normalized

    match = LETTER_PREFIX_PATTERN.match(candidate)
    if match:
        normalized = normalize_option_letter(match.group(1))
        if normalized is not None and option_letter_to_index(normalized) is not None and option_letter_to_index(normalized) < len(sample.options):
            return normalized

    lowered = candidate.casefold()
    for idx, option in enumerate(sample.options):
        if lowered == option.casefold():
            return option_index_to_letter(idx)
    for idx, option in enumerate(sample.options):
        if option.casefold() in lowered or lowered in option.casefold():
            return option_index_to_letter(idx)
    return None


def parse_answer_prediction(sample: QuestionSample, payload: str | dict[str, object]) -> AnswerPrediction:
    """Parse a JSON answer or a bare option letter; raises ValueError when no valid option is found."""
    raw_text = payload if isinstance(payload, str) else None
    if isinstance(payload, str):
        try:
            payload = extract_json_object(payload)
        except ValueError:
            selected_option = option_letter_for_text(sample, raw_text)
            if selected_option is None:
                raise
            selected_option_text = option_text_for_letter(sample, selected_option)
            return AnswerPrediction(
                selected_option=selected_option,
                selected_option_text=selected_option_text,
                confidence=0.0,
                reasoning=raw_text.strip() or "No reasoning provided.",
            )

    selected_option = normalize_option_letter(str(payload.get("selected_option", "")))
    selected_option_text = str(payload.get("selected_option_text") or "").strip()
    if option_text_for_letter(sample, selected_option) is None:
        selected_option = None

    if selected_option is None:
        selected_option = option_letter_for_text(sample, selected_option_text)
    if selected_option is None and raw_text is not None:
        selected_option = option_letter_for_text(sample, raw_text)
    if selected_option is None:
        raise ValueError("Model output did not contain a valid option letter.")

    confidence = float(payload.get("confidence", 0.0))
    reasoning = str(payload.get("reasoning", "")).strip() or "No reasoning provided."
    # The option letter is the metric-bearing prediction. Always derive the
    # accompanying text from it so a plausible but contradictory free-form
    # string cannot make the saved prediction internally inconsistent.
    selected_option_text = option_text_for_letter(sample, selected_option) or selected_option_text

    return AnswerPrediction(
        selected_option=selected_option,
        selected_option_text=selected_option_text,
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=reasoning,
    )


def fallback_answer_prediction(sample: QuestionSample, raw_outputs: list[str] | None = None) -> AnswerPrediction:
    """A deterministic per-question guess used when every answer attempt fails to parse."""
    seed = f"{sample.dataset_name}::{sample.video_id}::{sample.question_uid}"
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    option_index = int.from_bytes(digest[:4], "big") % max(len(sample.options), 1)
    selected_option = option_index_to_letter(option_index) or "A"
    raw_suffix = ""
    if raw_outputs:
        last = raw_outputs[-1].strip()
        if last:
            raw_suffix = f" Last raw model output: {last[:240]}"
    return AnswerPrediction(
        selected_option=selected_option,
        selected_option_text=option_text_for_letter(sample, selected_option),
        confidence=0.0,
        reasoning=(
            f"Fallback guess used after repeated unparsable answer outputs."
            f"{raw_suffix}"
        ),
    )


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


@dataclass(slots=True)
class AdaptiveTemporalProposal:
    start_sec: float
    end_sec: float
    label: str = ""
    confidence: float = 0.0
    rationale: str = ""
    source: str = "overview"
    start_time: str | None = None
    end_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdaptiveBudgetPolicy:
    local_evidence_needed: bool = True
    overview_min_ratio: float = 0.65
    local_budget_ratio: float = 0.35
    uncertainty: float = 1.0
    policy_name: str | None = None
    local_detail_level: str | None = None
    needs_global_context_for_refinement: bool | None = None
    # Explicit split; when None the allocation uses overview_min_ratio.
    overview_ratio: float | None = None
    local_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdaptiveOverviewPlan:
    queries: list[str]
    temporal_proposals: list[AdaptiveTemporalProposal] = field(default_factory=list)
    budget_policy: AdaptiveBudgetPolicy = field(default_factory=AdaptiveBudgetPolicy)
    reasoning_chain: list[str] = field(default_factory=list)
    rationale: str = ""
    local_retrieval_needed: bool | None = None
    local_retrieval_reason: str = ""
    evidence_needs: list[str] = field(default_factory=list)
    parse_failed: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["temporal_proposals"] = [proposal.to_dict() for proposal in self.temporal_proposals]
        payload["budget_policy"] = self.budget_policy.to_dict()
        return payload


RefinementAction = Literal["keep", "remove", "adjust", "merge"]


@dataclass(slots=True)
class AdaptiveRefinementAction:
    action: RefinementAction
    region_id: str | None = None
    region_ids: list[str] = field(default_factory=list)
    start_sec: float | None = None
    end_sec: float | None = None
    new_label: str | None = None
    rationale: str | None = None
    confidence: float | None = None
    # Per-region checks: does the clip show what its query describes, and does
    # it help answer the question (yes/partly/no/unclear).
    query_match: str | None = None
    helpful: str | None = None
    # How much a region helps (relevance), separate from how sure the refiner
    # is about the chosen action (confidence).
    relevance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdaptiveMemoryRefinement:
    actions: list[AdaptiveRefinementAction] = field(default_factory=list)
    uncertainty: float = 1.0
    reasoning_chain: list[str] = field(default_factory=list)
    parse_failed: bool = False
    fallback_reason: str | None = None
    # False when no region shows anything that helps answer.
    usable_local_evidence: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [action.to_dict() for action in self.actions]
        return payload


def fallback_overview_plan(queries: list[str], *, reason: str) -> AdaptiveOverviewPlan:
    return AdaptiveOverviewPlan(
        queries=[query for query in queries if str(query).strip()],
        temporal_proposals=[],
        budget_policy=AdaptiveBudgetPolicy(
            local_evidence_needed=True,
            overview_min_ratio=0.70,
            local_budget_ratio=0.30,
            uncertainty=1.0,
            policy_name="GLOBAL_HEAVY",
            local_detail_level="STANDARD",
        ),
        reasoning_chain=[],
        parse_failed=True,
        fallback_reason=reason,
    )


def fallback_memory_refinement(*, reason: str) -> AdaptiveMemoryRefinement:
    return AdaptiveMemoryRefinement(actions=[], uncertainty=1.0, reasoning_chain=[], parse_failed=True, fallback_reason=reason)


def _payload_from_input(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        payload = extract_json_object(payload)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


# Overview / local share of the final visual budget for each planner policy.
_POLICY_RATIOS = {
    "GLOBAL_ONLY": (1.0, 0.0),
    "GLOBAL_HEAVY": (0.70, 0.30),
    "BALANCED": (0.50, 0.50),
    "LOCAL_HEAVY": (0.30, 0.70),
    "LOCAL_ONLY": (0.0, 1.0),
}
_MAX_LOCAL_RETRIEVAL_QUERIES = 3


def force_allocation_policy(plan: AdaptiveOverviewPlan, policy_name: str) -> dict[str, Any]:
    """Replace the planner's allocation policy with ``policy_name`` (``adaptive_allocation: false``).

    Only the global/local split changes; the plan's queries, proposals and retrieval decision stay.
    """

    if policy_name not in _POLICY_RATIOS:
        raise ValueError(f"Unsupported allocation policy: {policy_name!r}")
    original_policy = plan.budget_policy.to_dict()
    overview_ratio, local_ratio = _POLICY_RATIOS[policy_name]
    plan.budget_policy.policy_name = policy_name
    plan.budget_policy.overview_min_ratio = overview_ratio
    plan.budget_policy.local_budget_ratio = local_ratio
    plan.budget_policy.overview_ratio = overview_ratio
    plan.budget_policy.local_ratio = local_ratio
    plan.budget_policy.local_evidence_needed = local_ratio > 0.0
    return {
        "fixed_policy_name": policy_name,
        "planner_policy_name": original_policy.get("policy_name"),
        "planner_overview_ratio": original_policy.get("overview_ratio"),
        "planner_local_ratio": original_policy.get("local_ratio"),
        "planner_local_retrieval_needed": plan.local_retrieval_needed,
    }
_LOCAL_DETAIL_LEVELS = {
    "NONE",
    "COVERAGE",
    "STANDARD",
    "FINE",
}


def _bool_from_payload(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return bool(value)


def _parse_time_string(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.lower().removeprefix("timestamp").removeprefix("time").strip()
    text = text.removesuffix("seconds").removesuffix("second").removesuffix("secs").removesuffix("sec").removesuffix("s").strip()
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return None


def _parse_temporal_bounds(item: dict[str, Any]) -> tuple[float, float, str | None, str | None] | None:
    start_time = str(item["start_time"]).strip() if item.get("start_time") not in (None, "") else None
    end_time = str(item["end_time"]).strip() if item.get("end_time") not in (None, "") else None
    start_sec = _parse_time_string(start_time) if start_time is not None else None
    end_sec = _parse_time_string(end_time) if end_time is not None else None
    if start_sec is None and "start_sec" in item:
        start_sec = _parse_time_string(item.get("start_sec"))
    if end_sec is None and "end_sec" in item:
        end_sec = _parse_time_string(item.get("end_sec"))
    if start_sec is None or end_sec is None:
        return None
    return start_sec, end_sec, start_time, end_time


def parse_overview_plan(
    payload: str | dict[str, Any],
    *,
    duration_sec: float | None = None,
) -> AdaptiveOverviewPlan:
    """Parse a planner reply; raises ValueError (with a message for the retry prompt) when it breaks the schema."""
    data = _payload_from_input(payload)
    required_top_keys = {"rationale", "budget_policy", "local_retrieval"}
    missing_top_keys = required_top_keys - set(data)
    if missing_top_keys:
        raise ValueError(f"Adaptive overview plan missing required keys: {sorted(missing_top_keys)}.")

    local_retrieval_payload = data.get("local_retrieval", {}) or {}
    if not isinstance(local_retrieval_payload, dict):
        local_retrieval_payload = {}

    policy_payload = data.get("budget_policy", {}) or {}
    if not isinstance(policy_payload, dict):
        policy_payload = {}

    local_retrieval_needed = _bool_from_payload(
        local_retrieval_payload.get("needed", policy_payload.get("local_evidence_needed")),
        True,
    )

    queries_payload = local_retrieval_payload.get("queries", data.get("queries", []))
    if not isinstance(queries_payload, list):
        raise ValueError("Expected 'queries' to be a list.")
    queries = [str(query).strip() for query in queries_payload if str(query).strip()]
    if not local_retrieval_needed and queries:
        raise ValueError("Expected local_retrieval.queries to be empty when local_retrieval.needed=false.")
    if local_retrieval_needed and not queries:
        raise ValueError("Adaptive overview plan did not contain any queries.")
    queries = queries[:_MAX_LOCAL_RETRIEVAL_QUERIES]

    proposals: list[AdaptiveTemporalProposal] = []
    for item in data.get("temporal_proposals", []) or []:
        if not isinstance(item, dict):
            continue
        bounds = _parse_temporal_bounds(item)
        if bounds is None:
            continue
        start_sec, end_sec, start_time, end_time = bounds
        if duration_sec is not None:
            start_sec = max(0.0, min(float(duration_sec), start_sec))
            end_sec = max(0.0, min(float(duration_sec), end_sec))
        if end_sec <= start_sec:
            continue
        if "source" not in item:
            continue
        source = str(item["source"]).strip().lower()
        if source not in {"question", "overview"}:
            continue
        default_confidence = 1.0 if source == "question" else 0.5
        proposals.append(
            AdaptiveTemporalProposal(
                start_sec=start_sec,
                end_sec=end_sec,
                label=str(item.get("label", "")).strip(),
                confidence=_clamp01(item.get("confidence"), default_confidence),
                rationale=str(item.get("rationale", "")).strip(),
                source=source,
                start_time=start_time,
                end_time=end_time,
            )
        )

    policy_name_payload = policy_payload.get("policy_name")
    policy_name = str(policy_name_payload).strip().upper() if policy_name_payload is not None else None
    if policy_name not in _POLICY_RATIOS:
        raise ValueError("Expected budget_policy.policy_name to be one of the allowed policy names.")
    overview_min_ratio, local_budget_ratio = _POLICY_RATIOS[policy_name]
    raw_local_detail_level = policy_payload.get("local_detail_level")
    local_detail_level = (
        str(raw_local_detail_level).strip().upper()
        if raw_local_detail_level is not None
        else None
    )
    if local_detail_level and local_detail_level not in _LOCAL_DETAIL_LEVELS:
        raise ValueError(
            "Expected budget_policy.local_detail_level to be one of "
            f"{sorted(_LOCAL_DETAIL_LEVELS)}."
        )
    budget_policy = AdaptiveBudgetPolicy(
        local_evidence_needed=bool(local_budget_ratio > 0.0 or proposals),
        overview_min_ratio=overview_min_ratio,
        local_budget_ratio=local_budget_ratio,
        uncertainty=_clamp01(policy_payload.get("uncertainty"), 0.0),
        policy_name=policy_name,
        local_detail_level=local_detail_level,
        needs_global_context_for_refinement=(
            _bool_from_payload(policy_payload.get("needs_global_context_for_refinement"), False)
            if "needs_global_context_for_refinement" in policy_payload
            else None
        ),
    )
    reasoning_chain = [str(step).strip() for step in data.get("reasoning_chain", []) if str(step).strip()] if isinstance(data.get("reasoning_chain", []), list) else []
    rationale = str(data.get("rationale", "")).strip()
    if rationale and not reasoning_chain:
        reasoning_chain = [rationale]
    evidence_needs_payload = local_retrieval_payload.get("evidence_needs", [])
    evidence_needs = (
        [str(need).strip() for need in evidence_needs_payload if str(need).strip()]
        if isinstance(evidence_needs_payload, list)
        else []
    )
    return AdaptiveOverviewPlan(
        queries=queries,
        temporal_proposals=proposals,
        budget_policy=budget_policy,
        reasoning_chain=reasoning_chain,
        rationale=rationale,
        local_retrieval_needed=local_retrieval_needed,
        local_retrieval_reason=str(local_retrieval_payload.get("reason", "")).strip(),
        evidence_needs=evidence_needs,
    )


def _strict_refinement_payload(payload: str | dict[str, Any]) -> dict[str, Any]:
    data = extract_outer_json_object(payload) if isinstance(payload, str) else payload
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    if not isinstance(data.get("actions"), list):
        raise ValueError("Expected a top-level 'actions' list.")
    return data


_V6_CHECK_VALUES = {"yes", "partly", "no", "unclear"}


def _v6_check(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in _V6_CHECK_VALUES else "unclear"


def _optional_unit_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        float(value)
    except (TypeError, ValueError):
        return None
    return _clamp01(value, 0.0)


def parse_memory_refinement_v6(payload: str | dict[str, Any]) -> AdaptiveMemoryRefinement:
    """Parse a refiner reply (strict: one complete JSON object with an ``actions`` list).

    Each action names one region with its `query_match`/`helpful` checks,
    `relevance`, and `confidence`. A merge is written on one region with the
    other members in `merge_with`; it becomes a merge action over all of them,
    with its span left to the caller.
    """

    data = _strict_refinement_payload(payload)
    actions: list[AdaptiveRefinementAction] = []
    for item in data["actions"]:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip().lower()
        region_id = str(item.get("region_id", "")).strip()
        if action not in {"keep", "adjust", "merge", "remove"} or not region_id:
            continue
        start_sec = end_sec = None
        region_ids: list[str] = []
        if action == "adjust":
            try:
                start_sec = float(item.get("start_sec"))
                end_sec = float(item.get("end_sec"))
            except (TypeError, ValueError):
                continue
        if action == "merge":
            merge_with = item.get("merge_with")
            others = (
                [str(other).strip() for other in merge_with if str(other).strip() and str(other).strip() != region_id]
                if isinstance(merge_with, list)
                else []
            )
            if not others:
                continue
            region_ids = list(dict.fromkeys([region_id, *others]))
        actions.append(
            AdaptiveRefinementAction(
                action=action,  # type: ignore[arg-type]
                region_id=region_id,
                region_ids=region_ids,
                start_sec=start_sec,
                end_sec=end_sec,
                rationale=str(item.get("rationale", "")).strip() or None,
                confidence=_optional_unit_float(item.get("confidence")),
                query_match=_v6_check(item.get("query_match")),
                helpful=_v6_check(item.get("helpful")),
                relevance=_optional_unit_float(item.get("relevance")),
            )
        )
    notes = data.get("notes", [])
    usable = data.get("usable_local_evidence")
    return AdaptiveMemoryRefinement(
        actions=actions,
        uncertainty=_clamp01(data.get("uncertainty"), 1.0),
        reasoning_chain=[str(step).strip() for step in notes if str(step).strip()] if isinstance(notes, list) else [],
        usable_local_evidence=None if usable is None else _bool_from_payload(usable, True),
    )
