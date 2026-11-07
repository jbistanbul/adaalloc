"""Prompts for the planner, the refiner and the final answer, and the frame metadata they show."""

from __future__ import annotations

import json
from typing import Any, Iterable

from .data import FrameReference, QuestionSample, RetrievedSegment, VideoEvidenceStream
from .planning import option_restated_by_query


def render_frame_evidence(frames: list[FrameReference]) -> str:
    """One metadata line per frame, in the order the frames are given to the model."""
    if not frames:
        return "No frames selected."
    rendered: list[str] = []
    for position, frame in enumerate(frames, start=1):
        metadata_parts = []
        if frame.evidence_type:
            metadata_parts.append(f"evidence_type={frame.evidence_type}")
        if frame.evidence_role:
            metadata_parts.append(f"role={frame.evidence_role}")
        if frame.evidence_region:
            metadata_parts.append(f"region={frame.evidence_region}")
        metadata = f" {' '.join(metadata_parts)}" if metadata_parts else ""
        rendered.append(
            f"- provided_frame={position} frame={frame.frame_index} timestamp={frame.timestamp_sec:.2f}s{metadata}"
        )
    return "\n".join(rendered)


def render_video_stream_evidence(streams: list[VideoEvidenceStream]) -> str:
    """Render frame metadata in the same grouping and order as native video inputs."""

    if not streams:
        return "No video streams selected."

    rendered: list[str] = []
    for stream_position, stream in enumerate(streams, start=1):
        timestamps = [
            float(frame.timestamp_sec)
            for frame in stream.frame_references[: len(stream.frames)]
        ]
        source_range = (
            f"{min(timestamps):.2f}s..{max(timestamps):.2f}s"
            if timestamps
            else "unknown"
        )
        rendered.append(
            f"- video_stream={stream_position} label={stream.label!r} "
            f"kind={stream.kind} frame_count={len(stream.frames)} "
            f"source_time_range={source_range}"
        )
        for frame_position in range(len(stream.frames)):
            if frame_position >= len(stream.frame_references):
                rendered.append(
                    f"  - stream_frame={frame_position + 1} source_frame=unknown "
                    "absolute_timestamp=unknown"
                )
                continue
            frame = stream.frame_references[frame_position]
            metadata_parts = []
            if frame.evidence_type:
                metadata_parts.append(f"evidence_type={frame.evidence_type}")
            if frame.evidence_role:
                metadata_parts.append(f"role={frame.evidence_role}")
            if frame.evidence_region:
                metadata_parts.append(f"region={frame.evidence_region}")
            metadata = f" {' '.join(metadata_parts)}" if metadata_parts else ""
            rendered.append(
                f"  - stream_frame={frame_position + 1} source_frame={frame.frame_index} "
                f"absolute_timestamp={frame.timestamp_sec:.2f}s{metadata}"
            )
    return "\n".join(rendered)


def _render_options(options: Iterable[str]) -> str:
    rendered = []
    for idx, option in enumerate(options):
        rendered.append(f"{chr(ord('A') + idx)}. {option}")
    return "\n".join(rendered)


def _render_option_letters(options: Iterable[str]) -> str:
    letters = [chr(ord("A") + idx) for idx, _ in enumerate(options)]
    return ", ".join(letters) if letters else "the option letters shown above"


def _render_temporal_coverage_summary(retrieved_segments: list[RetrievedSegment]) -> str:
    """Render final-answer temporal metadata without leaking retrieval query text.

    Query text, retrieval scores, and source labels are useful for diagnostics
    and memory refinement, but they can bias the final answer model into
    treating retrieval metadata as visual proof. The final prompt only needs
    timing/coverage hints.
    """

    return "\n".join(
        f"- span=[{segment.start_sec:.2f}, {segment.end_sec:.2f}]"
        for segment in retrieved_segments
    ) or "- none"


def _render_video_info(sample: QuestionSample) -> str:
    duration = sample.video_metadata.duration_sec
    fps = sample.video_metadata.fps
    total_frames = sample.video_metadata.total_frames
    parts: list[str] = []
    parts.append(f"- Duration: {duration:.1f} seconds" if duration is not None else "- Duration: unknown")
    parts.append(f"- FPS: {fps:.3f}" if fps is not None else "- FPS: unknown")
    parts.append(f"- Total Frames: {total_frames}" if total_frames is not None else "- Total Frames: unknown")
    return "\n".join(parts)


def build_overview_planning_prompt(sample: QuestionSample, observed_frames: str) -> str:
    """Planner prompt: choose the allocation policy, retrieval queries and temporal proposals from the overview."""

    def example_budget_policy(policy_name: str, *, needs_global_context_for_refinement: bool) -> dict[str, Any]:
        return {
            "policy_name": policy_name,
            "needs_global_context_for_refinement": needs_global_context_for_refinement,
        }

    def render_example(
        title: str,
        user_question: str,
        options: list[str],
        observed_overview: str,
        output: dict[str, Any],
    ) -> str:
        return "\n".join(
            [
                title,
                "User Question:",
                user_question,
                "",
                "Options:",
                "\n".join(f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(options)),
                "",
                "Observed Overview:",
                observed_overview,
                "",
                "Output:",
                json.dumps(output, indent=2, ensure_ascii=False),
            ]
        )

    examples_text = "\n\n".join(
        [
            render_example(
                "Example 1 — explicit original-video interval, inspect directly:",
                "What did I put in the fridge from 00:18 to 00:21?",
                ["the leftovers", "the eggs", "the vegetables", "sauce"],
                "Overview shows a kitchen workflow, but the exact object is too small/sparse to identify globally.",
                {
                    "rationale": "The question gives the exact original-video interval, so inspect it directly without UniversalVTG search.",
                    "budget_policy": example_budget_policy("LOCAL_ONLY", needs_global_context_for_refinement=False),
                    "local_retrieval": {
                        "needed": False,
                        "reason": "The relevant interval is already specified by the question.",
                        "evidence_needs": ["inspect the object placed in the fridge between 18 and 21 seconds"],
                        "queries": [],
                    },
                    "temporal_proposals": [
                        {
                            "start_sec": 18.0,
                            "end_sec": 21.0,
                            "source": "question",
                            "label": "specified fridge interval",
                            "rationale": "The question explicitly gives this original-video span.",
                        }
                    ],
                },
            ),
            render_example(
                "Example 2 — ordering needs retrieval anchors but remains global-heavy:",
                "Arrange these events in chronological order: (1) woman tapes her hands; (2) woman boxes in a ring; (3) woman does sit-ups on a beach; (4) pictures of woman in a bikini are shown.",
                ["2->1->3->4", "3->2->1->4", "4->3->2->1", "1->2->3->4"],
                "Overview shows several activity phases, but not enough frames to confidently order all listed events.",
                {
                    "rationale": "The task is mainly temporal ordering, so keep global context while retrieving local anchors.",
                    "budget_policy": example_budget_policy("GLOBAL_HEAVY", needs_global_context_for_refinement=True),
                    "local_retrieval": {
                        "needed": True,
                        "reason": "UniversalVTG should localize event anchors; final ordering still needs broad context.",
                        "evidence_needs": [
                            "localize the hand-taping event",
                            "localize the boxing event",
                            "localize the beach sit-up and bikini-picture scenes",
                        ],
                        "queries": [
                            "The woman taped her hands.",
                            "The woman boxed in a ring.",
                            "The woman did sit-ups on a beach.",
                        ],
                    },
                    "temporal_proposals": [],
                },
            ),
            render_example(
                "Example 3 — count question, decompose into object/action/count visibility:",
                "How many sticks does the protagonist put in the incense burner?",
                ["3", "2", "5", "1"],
                "Overview shows a ceremonial indoor scene but not the small sticks or exact placement moment.",
                {
                    "rationale": "The answer requires counting small items during a localized action, so retrieve neutral evidence elements without asserting a number.",
                    "budget_policy": example_budget_policy("LOCAL_HEAVY", needs_global_context_for_refinement=True),
                    "local_retrieval": {
                        "needed": True,
                        "reason": "UniversalVTG should localize the burner, placement action, and countable stick view.",
                        "evidence_needs": ["find the incense burner", "inspect the stick placement action", "count visible sticks"],
                        "queries": [
                            "The incense burner was shown.",
                            "The protagonist placed sticks into the incense burner.",
                            "The sticks in the incense burner were clearly visible.",
                        ],
                    },
                    "temporal_proposals": [],
                },
            ),
            render_example(
                "Example 4 — answer options are candidate outcomes; retrieve neutral anchors instead:",
                "What does the protagonist do when the man with the gun keeps approaching her?",
                [
                    "The protagonist cuts off the hand of the man with the gun with her sword",
                    "The protagonist cuts off two fingers of the man with the gun with her sword",
                    "The protagonist impales the man with the gun",
                    "The protagonist breaks the windows with her sword and flees",
                ],
                "Overview shows a confrontation, but the exact defensive action and visible result are unclear.",
                {
                    "rationale": "The options contain candidate answers, so retrieval should localize the confrontation and visible result without asserting any option.",
                    "budget_policy": example_budget_policy("LOCAL_HEAVY", needs_global_context_for_refinement=True),
                    "local_retrieval": {
                        "needed": True,
                        "reason": "UniversalVTG should find the approach, reaction, and visible outcome.",
                        "evidence_needs": [
                            "localize when the man with the gun approaches",
                            "inspect the protagonist's reaction",
                            "inspect the visible outcome of the confrontation",
                        ],
                        "queries": [
                            "The man with the gun approached the protagonist.",
                            "The person reacted as the man with the gun approached.",
                            "The person and the man with the gun are visible.",
                        ],
                    },
                    "temporal_proposals": [],
                },
            ),
        ]
    )

    return f"""You are a visual evidence planner for video multiple-choice question answering.

Do NOT answer the question yet. Plan how to collect visual evidence for the final answer.
The sparse overview frames are a coarse global overview, not complete proof; decisive evidence may be between sampled frames.
Retrieval queries must find where to look, not assert which answer option is correct.

User Question:
{sample.question_text}

Options:
{_render_options(sample.options)}

Video Information:
{_render_video_info(sample)}

Observed Overview Frame Metadata:
{observed_frames}

---

Return a compact JSON plan that:
1. chooses the final global/local visual-budget policy;
2. decides whether UniversalVTG should search for unknown temporal regions;
3. if retrieval is needed, gives 1-3 answer-agnostic evidence-localization statements;
4. adds question- or overview-suggested candidate regions to temporal_proposals.

Visual budget meanings:
- GLOBAL/overview: sparse full-video coverage for story, order, scene changes, repeated events, summaries, broad context.
- LOCAL/detail: denser/higher-resolution frames from selected regions for small objects, text, colors, counts, spatial relations, short actions, or exact moments.

Budget Policy Choices:
- GLOBAL_ONLY: broad overview only. Ratio = 1.0 overview / 0.0 local.
- GLOBAL_HEAVY: mostly global context, limited local detail. Ratio = 0.70 overview / 0.30 local.
- BALANCED: both global context and local evidence, with an equal global/local split. Ratio = 0.50 overview / 0.50 local.
- LOCAL_HEAVY: fine-grained local evidence is required, but global context remains substantial. Ratio = 0.30 overview / 0.70 local.
- LOCAL_ONLY: rare; localized fine detail dominates and broad context is unlikely to help. Ratio = 0.0 overview / 1.0 local.

Choose only policy_name. Do not output numeric ratios in this variant.


Policy calibration:
- Retrieval need is separate from allocation policy: UniversalVTG may be useful while final budget remains GLOBAL_HEAVY or BALANCED.
- Default uncertain cases to BALANCED rather than LOCAL_HEAVY.
- Use GLOBAL_ONLY/GLOBAL_HEAVY for summary, ordering, broad progression, scene-level, repeated-event, or whole-video questions.
- Use BALANCED when both global context and localized anchors matter, including before/after/when questions.
- Use LOCAL_HEAVY only when fine-grained local inspection is required to distinguish options: small objects, visible text, colors, precise counts, spatial relations, brief interactions, or short localized actions.
- Use LOCAL_ONLY rarely: exact/sufficient video interval, or a single localized detail where broad context is unlikely to help.
- If repeated or similar events may be confused, preserve more global context.

UniversalVTG retrieval:
- local_retrieval.needed=true only when UniversalVTG should search for unknown regions.
- Set needed=false when overview is sufficient, the question gives an exact/sufficient original-video interval, or temporal_proposals already cover the candidate region.
- If needed=false, queries must be []. evidence_needs may still describe what to inspect directly.
- If needed=true, produce 1-3 short declarative, past-tense, visually observable, temporally localizable statements.

Answer-option use:
- Answer options are not evidence; use them only to infer what visual distinction must be resolved.
- Do not copy, paraphrase, convert, or assert option values in retrieval queries unless the value is also stated in the question or clearly visible in the overview frames.
- Generate retrieval queries from the original question and observed overview evidence, not from guessing which option may be correct.
- If options are locations, colors, counts, objects, people, actions, text, or outcomes, retrieve the evidence needed to identify the correct one rather than querying candidate answers.

Query construction rules:
- Build queries from the question subject, action, object, event, temporal anchor, and overview-observed context.
- Decompose into one evidence element per query.
- Query evidence slots, not answer choices: object visibility, action occurrence, location/attribute/count visibility, before/after anchor events, final visible state, or visible outcome.
- Do not copy/paraphrase/assert an option as a query.
- Do not generate one query per option.
- Do not assert option-only values (candidate color, number, place, identity, instrument, action, text, or outcome) unless they are also stated in the question or clearly supported by overview frames.
- Keep each query focused on one evidence element: main object/item, actor, anchor event, action, location/spatial relation, before/after anchor, visible text/number, countable items, attribute visibility, or visible outcome/state.
- Avoid hidden intent, motivation, absence-only claims, and multi-event story summaries.

Temporal proposals:
- Use source="question" only for absolute original-video times: video timestamp 88:07, MM:SS-MM:SS, or "120 seconds into the video".
- For explicit original-video intervals, set local_retrieval.needed=false, queries=[], and add the interval to temporal_proposals.
- Do NOT convert in-scene clocks, sports game clocks, match times, countdowns, dates, years, ages, jersey numbers, scores, rounds, periods, or quarters into video timestamps. Example: "10 minutes and 33 seconds in the second quarter" is a retrieval cue, not a video-time proposal.
- You may output numeric seconds (start_sec/end_sec) or timestamp strings (start_time/end_time); the system will normalize them.
- Use source="overview" for overview-suggested approximate regions. Do not invent precise spans; use broad spans or [] when uncertain.

Anti-hallucination:
- Use only nouns, verbs, roles, objects, and locations supported by the question, video metadata, or overview frames.
- Options may reveal the attribute type to inspect, but option-only answer values must not be asserted in retrieval queries.
- If a role is unnamed, use neutral terms like "the person", "the object", or "the item".
- The overview is incomplete; do not treat missing overview evidence as proof an event never happened.

Output requirements:
- JSON only; no markdown fences or commentary.
- Use exactly the top-level keys shown below.
- Do not output extra confidence or uncertainty fields.
- In model-ratio mode, do not output extra numeric self-assessment fields beyond overview_ratio/local_ratio.

STRICT OUTPUT FORMAT:
{{
  "rationale": "one short sentence explaining the evidence-collection plan",
  "budget_policy": {{
    "policy_name": "GLOBAL_ONLY|GLOBAL_HEAVY|BALANCED|LOCAL_HEAVY|LOCAL_ONLY",
    "needs_global_context_for_refinement": true
  }},
  "local_retrieval": {{
    "needed": true,
    "reason": "one short sentence explaining whether UniversalVTG retrieval is needed",
    "evidence_needs": ["specific visual fact to collect or inspect"],
    "queries": ["declarative UniversalVTG event statement"]
  }},
  "temporal_proposals": [
    {{
      "start_sec": 0.0,
      "end_sec": 10.0,
      "start_time": "00:00",
      "end_time": "00:10",
      "source": "question|overview",
      "label": "short scene/event label",
      "rationale": "why this span may matter"
    }}
  ]
}}

Examples:
{examples_text}
"""


def _region_payload(region: Any) -> dict[str, Any]:
    if hasattr(region, "to_dict"):
        return region.to_dict()
    if isinstance(region, dict):
        return region
    return {
        "region_id": getattr(region, "region_id", ""),
        "start_sec": getattr(region, "start_sec", 0.0),
        "end_sec": getattr(region, "end_sec", 0.0),
        "sources": getattr(region, "sources", []),
        "rationale": getattr(region, "rationale", ""),
        "candidate_details": getattr(region, "candidate_details", []),
    }


def _v6_query_ids(
    sample: QuestionSample,
    retrieved_segments: Iterable[RetrievedSegment] | None,
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Number retrieval queries Q0.. with the original question first."""

    seen: dict[tuple[str, str], None] = {}
    for segment in retrieved_segments or []:
        query_type = "original_question" if segment.source == "universal_vtg_original_question" else "generated"
        seen.setdefault((query_type, segment.query), None)
    for region in regions:
        for detail in region.get("candidate_details") or []:
            seen.setdefault((str(detail.get("query_type") or "generated"), str(detail.get("query") or "")), None)
    ordered = sorted(
        (key for key in seen if key[1]),
        key=lambda key: 0 if key[0] == "original_question" else 1,
    )
    queries = []
    for index, (query_type, text) in enumerate(ordered):
        restated = (
            None
            if query_type == "original_question"
            else option_restated_by_query(text, sample.options, sample.question_text)
        )
        queries.append(
            {
                "id": f"Q{index}",
                "text": text,
                "original": query_type == "original_question",
                "restates_option": chr(ord("A") + restated) if restated is not None else None,
            }
        )
    return queries


def build_memory_refinement_prompt_v6(
    sample: QuestionSample,
    memory_regions: Iterable[Any],
    observed_frames: str,
    retrieved_segments: Iterable[RetrievedSegment] | None = None,
    *,
    plan: Any | None = None,
    frame_counts_by_region: dict[str, int] | None = None,
    context_clip_count: int = 0,
    visual_input_mode: str = "video_streams",
) -> str:
    """Guarded refinement prompt: two checks and one action per region.

    Each region is judged on whether its clip shows what its query describes
    and whether it helps answer the question. Remove and merge are allowed
    but bounded by explicit conditions that the guarded refinement also
    enforces in code. Queries that restate an answer option are flagged so the
    refiner judges the described event rather than confirming the guess.
    """

    regions = [_region_payload(region) for region in memory_regions]
    queries = _v6_query_ids(sample, retrieved_segments, regions)
    query_id_by_key = {(not query["original"], query["text"]): query["id"] for query in queries}
    query_lines = [
        f"- {query['id']}{' (the original question)' if query['original'] else ''}: {query['text']}"
        + (f"  [restates option {query['restates_option']}: treat as a guess]" if query["restates_option"] else "")
        for query in queries
    ]
    region_lines = []
    for region in regions:
        ranks: dict[str, int] = {}
        for detail in region.get("candidate_details") or []:
            key = (str(detail.get("query_type") or "generated") != "original_question", str(detail.get("query") or ""))
            query_id = query_id_by_key.get(key)
            if query_id is not None:
                rank = int(detail.get("rank") or 0)
                ranks[query_id] = min(rank, ranks.get(query_id, rank))
        sources = list(region.get("sources") or [])
        planner_notes = [
            part.strip()
            for part in str(region.get("rationale") or "").split("; ")
            if part.strip() and not part.strip().startswith("UniversalVTG")
        ]
        origin = []
        if ranks:
            origin.append("found by " + ", ".join(f"{query_id} (rank {rank})" for query_id, rank in sorted(ranks.items())))
        if planner_notes and ("overview_proposal" in sources or "question_time_ref" in sources):
            origin.append(f'proposed by the planner: "{"; ".join(planner_notes)[:300]}"')
        if "question_time_ref" in sources:
            origin.append("covers a time named in the question")
        region_id = str(region.get("region_id"))
        frame_count = (frame_counts_by_region or {}).get(region_id)
        frame_text = f"{int(frame_count)} frames in its clip" if frame_count is not None else "frame count unknown"
        region_lines.append(
            f"- {region_id} [{float(region.get('start_sec', 0.0)):.1f}s-{float(region.get('end_sec', 0.0)):.1f}s], "
            f"{frame_text}; " + ("; ".join(origin) or "origin unknown")
        )
    budget_policy = getattr(plan, "budget_policy", None)
    evidence_needs = [str(need).strip() for need in (getattr(plan, "evidence_needs", None) or []) if str(need).strip()]
    needs_text = "\n".join(f"  - {need}" for need in evidence_needs) or "  - (none listed)"
    planner_rationale = str(getattr(plan, "rationale", "") or "").strip()[:600] or "(none)"
    duration = sample.video_metadata.duration_sec
    duration_text = f"{duration:.1f} seconds" if duration is not None else "unknown"
    media_intro = (
        "You are given one short, sparse clip per region."
        if visual_input_mode == "video_streams"
        else "You are given sparse frames sampled from the regions; each frame's timestamp tells you which region it belongs to."
    )
    context_note = (
        '\nClips labelled "Context frames outside every memory region" are not regions. '
        "Use them only as surrounding context and give them no action."
        if context_clip_count
        else ""
    )
    return f"""You are refining the temporal evidence memory for a multiple-choice video question.
Each memory region is a candidate time span, found by a retrieval query or proposed by the planner. {media_intro} Decide, region by region, what the final answerer should look at closely. Do not answer the question.

**Question:** {sample.question_text}

**Options:**
{_render_options(sample.options)}

**Video duration:** {duration_text}

**Planner's evidence plan**
- Allocation policy: {getattr(budget_policy, "policy_name", None) or "unknown"}
- Evidence needed:
{needs_text}
- Planner rationale: {planner_rationale}

**Retrieval queries**
{chr(10).join(query_lines) or "- none"}

**Memory regions** (one clip each, in this order)
{chr(10).join(region_lines) or "- none"}{context_note}

**Clip frames**
{observed_frames}

---

**Step 1: two checks per region.** Look only at the frames inside that region's own clip.
- `query_match`: does the clip show the event or object described by the query that found it? For a planner-proposed region, use the planner note instead. Answer yes, partly, no, or unclear.
- `helpful`: does the clip show something that helps answer the question: an item under "Evidence needed", a moment the question refers to, or a detail that tells the options apart? Answer yes, partly, no, or unclear.

A query marked as restating an option is a guess about the answer. Judge whether the clip shows the event it describes, not whether that option is correct. A clip that contradicts such a query can be the most useful evidence, so it can still be `helpful`.

**Step 2: exactly one action per region.**
- `keep`: the region is helpful, partly helpful, or unclear. Give `relevance` from 0 to 1: about 0.9 decisive, 0.7 useful but incomplete, 0.4 weak but plausible, 0.2 or less probably not needed. Spread the scores across regions; do not give every region the same value.
- `adjust`: the useful moment visibly fills only part of the region. Give a tighter `start_sec` and `end_sec` inside the region, at timestamps you actually saw. The original region stays as context.
- `merge`: this region and the regions in `merge_with` show one continuous event, such as a setup and its outcome split across neighbouring regions no more than about 15 seconds apart. Do not merge separate occurrences.
- `remove`: only when all of these hold: `query_match` is no, `helpful` is no, and the frames positively show something unrelated (say what they show instead). Not seeing the event in a sparse clip is not enough. Do not remove a region whose clip has fewer than 3 frames, or a region that covers a time named in the question.

For questions about order, first/last, before/after, counting, or repetition, keep every distinct occurrence and temporal anchor; remove only clear duplicates or clearly unrelated scenes.

**Step 3: overall verdict.** If no region shows anything that helps answer the question, set `usable_local_evidence` to false; the answer will then be made from a uniform overview of the whole video. Otherwise set it to true.

**Output.** Return JSON only, with `actions` first and one entry per region:
{{
  "actions": [
    {{"region_id": "r1", "query_match": "yes|partly|no|unclear", "helpful": "yes|partly|no|unclear", "action": "keep", "relevance": 0.0, "confidence": 0.0, "rationale": "<visible cue and timestamp>"}},
    {{"region_id": "r2", "query_match": "...", "helpful": "...", "action": "adjust", "start_sec": 0.0, "end_sec": 0.0, "relevance": 0.0, "confidence": 0.0, "rationale": "..."}},
    {{"region_id": "r3", "query_match": "...", "helpful": "...", "action": "merge", "merge_with": ["r4"], "relevance": 0.0, "confidence": 0.0, "rationale": "..."}},
    {{"region_id": "r5", "query_match": "no", "helpful": "no", "action": "remove", "relevance": 0.0, "confidence": 0.0, "rationale": "<what the clip shows instead>"}}
  ],
  "usable_local_evidence": true,
  "uncertainty": 0.0,
  "notes": ["at most three short steps"]
}}
`relevance` is how much the region helps answer the question. `confidence` is how sure you are about the action. `uncertainty` is your overall uncertainty about these decisions. Keep each rationale under 20 words. No markdown fences or extra text.
"""


def build_simple_answer_prompt(sample: QuestionSample) -> str:
    """Letter-only answer prompt for the uniform-frame answers."""
    return f"""Select the best answer to the following multiple-choice question based on the video.
Respond with only the letter ({_render_option_letters(sample.options)}) of the correct option.

Question: {sample.question_text}
{_render_options(sample.options)}

The best answer is:"""


def build_grounded_structured_answer_prompt(
    sample: QuestionSample,
    observed_frames: str,
    retrieved_segments: list[RetrievedSegment],
    *,
    video_streams: bool = False,
) -> str:
    """Final-answer prompt over the overview and detail streams; asks for a JSON answer."""
    evidence_layout = (
        """The visual evidence is supplied as multiple separately encoded video streams in the listed order.
The first stream is generally a broad overview and later streams are focused detail clips.
Each stream starts at model-relative time zero and is not temporally contiguous with the preceding stream.
Use the per-stream stream_frame, source_frame, and absolute_timestamp mapping below to reason on the original source timeline."""
        if video_streams
        else "The visual evidence is supplied as chronologically ordered raw video frames."
    )
    return f"""You are synthesizing the final answer to a multiple-choice video question.

{evidence_layout}
The frames are sampled visual evidence from retrieved temporal regions of the video.
They may include low-resolution overview evidence and higher-resolution focused evidence from retrieved temporal regions.
Frame metadata may label evidence_type=overview for broad low-resolution context and evidence_type=local_detail for focused high-resolution evidence from selected memory regions.
Treat the directly observed frames as the primary evidence.
Use retrieved segment spans only as supporting references for timing and coverage.
Temporal spans are only coverage hints; answer from the visible frames, not from the metadata or retrieval wording.

**User's Question:** {sample.question_text}

**Multiple Choice Options:**
{_render_options(sample.options)}

**Video Information:**
{_render_video_info(sample)}

**Evidence from Directly Observed Frames:**
{observed_frames}

**Temporal Coverage Metadata:**
{_render_temporal_coverage_summary(retrieved_segments)}

---

**Your Task:**
Select the single best option and provide a concise, evidence-grounded explanation.

**Grounded answering procedure**
1. Identify the visible cue(s) in the frames that are relevant to the question.
2. Compare the answer options against those visible cue(s); do not choose an option just because it appears in a query, timestamp, retrieval span, or metadata field.
3. Use overview frames to understand broad context, scene transitions, and temporal order; use local_detail frames for fine-grained objects, text, colors, counts, spatial relations, and brief actions.
4. When overview and local_detail frames cover the same or nearby timestamp, prefer local_detail frames for visual details and overview frames for context.
5. If evidence spans multiple sampled regions, synthesize the relevant before/after or setup/outcome relationship before selecting the option.
6. If the visible evidence is incomplete or ambiguous, still choose the best-supported option but lower confidence and avoid overclaiming.

**Guidelines**
1. **Select Option**: Choose exactly one option letter from the options shown above.
2. **Selected Option Text**: Must exactly match the text of the chosen option.
3. **Confidence**: Provide a confidence score between 0.0 and 1.0.
4. **Reasoning**: Briefly explain which visible evidence supports the answer and why the chosen option fits better than plausible alternatives.
5. **Temporal Alignment**: If the question is temporal ("before", "after", "when", exact times), explicitly align the explanation to the relevant visible part of the evidence.
6. **Evidence Discipline**: Do not invent unseen text, counts, identities, motives, or actions.

**Reasoning Style**
- Keep the reasoning concise and evidence-based.
- Mention the decisive visible cue and, when useful, the most informative timestamps or spans.
- Prefer the option that is directly supported over one that requires extra assumptions.
- If needed, briefly rule out the strongest competing option.

**STRICT OUTPUT FORMAT**
Return JSON only with exactly these keys:
- `selected_option`
- `selected_option_text`
- `confidence`
- `reasoning`

The `selected_option` must be one of the option letters shown above.
The `selected_option_text` must exactly match one of the option texts shown above.

**Example Response:**
{{
  "selected_option": "B",
  "selected_option_text": "Snowy",
  "confidence": 0.95,
  "reasoning": "Snow is visibly falling in the opening scene and white snow is already accumulated on the ground, which directly supports snowy weather."
}}

**Example Response (temporal reasoning):**
{{
  "selected_option": "C",
  "selected_option_text": "{sample.options[2] if len(sample.options) > 2 else (sample.options[0] if sample.options else 'Option C')}",
  "confidence": 0.82,
  "reasoning": "The confrontation happens first, and the next clearly visible action after that moment matches option C rather than the alternatives."
}}
"""
