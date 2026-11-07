"""Sample, evidence and result types, and the multiple-choice video QA dataset loader."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import DatasetConfig

OPTION_PATTERN = re.compile(r"^\(([A-Z])\)\s*(.*)$")


@dataclass(slots=True)
class VideoMetadata:
    total_frames: int | None = None
    fps: float | None = None
    duration_sec: float | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QuestionSample:
    dataset_name: str
    video_id: str
    question_uid: str
    question_text: str
    options: list[str]
    answer_letter: str | None
    answer_index: int | None
    video_path: str
    time_reference: str | None = None
    question_types: list[str] = field(default_factory=list)
    video_metadata: VideoMetadata = field(default_factory=VideoMetadata)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["video_metadata"] = self.video_metadata.to_dict()
        return payload


@dataclass(slots=True)
class FrameReference:
    frame_index: int
    timestamp_sec: float
    evidence_type: str = ""
    evidence_role: str = ""
    evidence_region: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VideoEvidenceStream:
    frames: list[Any]
    frame_references: list[FrameReference]
    kind: str
    label: str
    total_pixels: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VisualEvidenceInput:
    """The final or refinement input: several separately encoded video streams."""

    frames: list[Any]
    mode: str = "video"
    groups: list[str] = field(default_factory=list)
    group_total_pixels: dict[str, int] = field(default_factory=dict)
    video_streams: list[VideoEvidenceStream] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedSegment:
    query: str
    start_sec: float
    end_sec: float
    score: float
    frame_indices: list[int] = field(default_factory=list)
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerPrediction:
    selected_option: str
    confidence: float
    reasoning: str
    selected_option_text: str | None = None
    missing_information: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QuestionResult:
    sample: QuestionSample
    selected_frames: list[FrameReference]
    retrieved_segments: list[RetrievedSegment]
    generated_queries: list[str]
    prediction: AnswerPrediction
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "selected_frames": [frame.to_dict() for frame in self.selected_frames],
            "retrieved_segments": [segment.to_dict() for segment in self.retrieved_segments],
            "generated_queries": list(self.generated_queries),
            "prediction": self.prediction.to_dict(),
            "metadata": self.metadata,
        }


def sample_identity(sample: QuestionSample | dict[str, Any]) -> str:
    """Stable key of one question across runs, shards and workers."""

    if isinstance(sample, QuestionSample):
        return f"{sample.dataset_name}::{sample.video_id}::{sample.question_uid}"
    return f"{sample['dataset_name']}::{sample['video_id']}::{sample['question_uid']}"


def parse_question_and_options(raw_question: str) -> tuple[str, list[str]]:
    """Split ``question\n(A) ...\n(B) ...`` into the question stem and the ordered options."""

    lines = [line.strip() for line in raw_question.splitlines() if line.strip()]
    question_lines: list[str] = []
    options_by_letter: dict[str, str] = {}
    parsing_options = False
    for line in lines:
        match = OPTION_PATTERN.match(line)
        if match:
            parsing_options = True
            options_by_letter[match.group(1)] = match.group(2).strip()
            continue
        if parsing_options and options_by_letter:
            last_key = sorted(options_by_letter.keys())[-1]
            options_by_letter[last_key] = f"{options_by_letter[last_key]} {line}".strip()
        else:
            question_lines.append(line)
    ordered_options = [options_by_letter[key] for key in sorted(options_by_letter)]
    question_text = " ".join(question_lines).strip()
    return question_text or raw_question.strip(), ordered_options


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON list or JSON Lines file."""

    text = Path(path).read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _option_index(letter: str | None) -> int | None:
    if not letter:
        return None
    letter = str(letter).strip().upper()
    return ord(letter) - ord("A") if len(letter) == 1 and letter.isalpha() else None


def load_samples(config: DatasetConfig) -> list[QuestionSample]:
    """Load every question of the videos listed in ``video_ids``.

    Annotations are records ``{"key": video_id, "qa": [{"uid", "question", "answer", "question_type"?,
    "time_reference"?}], "video_info"?}``; ``video_paths`` maps video ids to files and ``video_info``
    maps video ids to ``{"fps", "frames", "duration_sec"}``.
    """

    video_paths = json.loads(Path(config.video_paths).read_text())
    video_ids = set(json.loads(Path(config.video_ids).read_text()))
    video_info = json.loads(Path(config.video_info).read_text()) if config.video_info else {}
    samples: list[QuestionSample] = []
    for record in _read_records(config.annotations):
        video_id = record["key"]
        if video_id not in video_ids:
            continue
        info = video_info.get(video_id) or video_info.get(f"{video_id}.mp4") or record.get("video_info") or {}
        fps = info.get("fps")
        total_frames = info.get("frames") or info.get("num_frames")
        duration_sec = info.get("duration_sec")
        if duration_sec is None and fps and total_frames:
            duration_sec = total_frames / fps
        source = "annotations" if (video_info.get(video_id) or video_info.get(f"{video_id}.mp4")) else ("record" if info else "unavailable")
        metadata = VideoMetadata(total_frames=total_frames, fps=fps, duration_sec=duration_sec, source=source)
        for qa in record.get("qa", []):
            question_text, options = parse_question_and_options(qa["question"])
            answer_letter = qa.get("answer")
            samples.append(
                QuestionSample(
                    dataset_name=config.name,
                    video_id=video_id,
                    question_uid=str(qa["uid"]),
                    question_text=question_text,
                    options=options,
                    answer_letter=answer_letter,
                    answer_index=_option_index(answer_letter),
                    video_path=str(video_paths[video_id]),
                    time_reference=qa.get("time_reference"),
                    question_types=list(qa.get("question_type", [])),
                    video_metadata=VideoMetadata(**metadata.to_dict()),
                )
            )
    return samples
