"""Write the annotation files that ``configs/datasets/*.yaml`` read from the official benchmark releases.

Each subcommand reads a benchmark's official annotations and its video directory and writes
``video_qa.json`` (JSON Lines, questions in the official order), ``video_ids.json``,
``video_paths.json`` and ``video_info.json`` to ``<data_root>/<benchmark>/annotations``
(``MLVU/annotations_dev`` for MLVU-Dev)::

    python scripts/prepare_data.py lvbench --annotations video_info.meta.jsonl --videos VIDEOS --data-root DATA
    python scripts/prepare_data.py mlvu_test --annotations test-ground-truth/test_mcq_gt.json --videos VIDEOS --data-root DATA
    python scripts/prepare_data.py mlvu_dev --annotations MLVU/json --videos MLVU/video --data-root DATA
    python scripts/prepare_data.py videomme --annotations videomme/test-00000-of-00001.parquet --videos VIDEOS --data-root DATA
    python scripts/prepare_data.py longvideobench --annotations lvb_val.json --videos VIDEOS --data-root DATA

``video_info.json`` gives every video's frame count, fps and duration, read as for the paper runs:
OpenCV's frame count and fps (rounded to three decimals) for LVBench, MLVU-Test and Video-MME, decord
for MLVU-Dev, and ffprobe for LongVideoBench. Video-MME needs pyarrow to read its parquet file and
LongVideoBench needs ``ffprobe`` (FFmpeg) on the PATH.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

# MLVU-Dev multiple-choice task files and their task names; sub-scene captioning and summarization are
# generation tasks.
MLVU_DEV_TASKS = {
    "1_plotQA": "plotQA",
    "2_needle": "needleQA",
    "3_ego": "ego",
    "4_count": "count",
    "5_order": "order",
    "6_anomaly_reco": "anomaly_reco",
    "7_topic_reasoning": "topic_reasoning",
}
# Official LongVideoBench fields kept with each question.
LONGVIDEOBENCH_QA_FIELDS = (
    "question_wo_referring_query", "correct_choice", "position", "topic_category", "question_category", "level",
    "subtitle_path", "starting_timestamp_for_subtitles", "duration", "view_count",
)
OUTPUT_DIRS = {
    "lvbench": "LVBench/annotations",
    "mlvu_test": "MLVU/annotations",
    "mlvu_dev": "MLVU/annotations_dev",
    "videomme": "Video-MME/annotations",
    "longvideobench": "LongVideoBench/annotations",
}


def _letter(index: int) -> str:
    return chr(ord("A") + index)


def _question(question: str, options: list[Any]) -> str:
    return "\n".join([question] + [f"({_letter(i)}) {option}" for i, option in enumerate(options)])


# ----------------------------------------------------------------------------------------------------
# Video metadata
# ----------------------------------------------------------------------------------------------------


def probe_opencv(path: str) -> dict[str, Any]:
    """OpenCV's frame count and fps, the fps rounded to three decimals."""

    import cv2

    capture = cv2.VideoCapture(path)
    frames, fps = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), round(capture.get(cv2.CAP_PROP_FPS), 3)
    capture.release()
    if not frames or not fps:
        raise RuntimeError(f"OpenCV cannot read {path}")
    return {"frames": frames, "fps": fps, "duration_sec": frames / fps}


def probe_decord(path: str) -> dict[str, Any]:
    """decord's frame count and average fps."""

    import decord

    reader = decord.VideoReader(path, num_threads=1)
    frames, fps = len(reader), float(reader.get_avg_fps())
    return {"frames": frames, "fps": fps, "duration_sec": frames / fps}


def probe_ffprobe(path: str) -> dict[str, Any]:
    """ffprobe's stream duration and frame rate (six decimals); frames from the container, else duration x fps."""

    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
               "-show_entries", "stream=nb_frames,avg_frame_rate,duration", path]
    stream = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)["streams"][0]
    fps = round(float(Fraction(stream["avg_frame_rate"])), 6)
    duration = float(stream["duration"])
    frames = int(stream["nb_frames"]) if "nb_frames" in stream else round(duration * fps)
    return {"frames": frames, "fps": fps, "duration_sec": duration}


def probe_videos(paths: dict[str, Path], probe: Callable[[str], dict[str, Any]], workers: int) -> dict[str, dict[str, Any]]:
    """``{video_id: {"frames", "fps", "duration_sec"}}`` for every video, probed in parallel."""

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} videos are missing, e.g. {missing[:3]}")
    with ProcessPoolExecutor(workers) as pool:
        return dict(zip(paths, pool.map(probe, [str(path) for path in paths.values()])))


# ----------------------------------------------------------------------------------------------------
# Benchmarks: each returns the video_qa records, the video files and their metadata
# ----------------------------------------------------------------------------------------------------


def lvbench(annotations: Path, videos: Path, workers: int):
    """``video_info.meta.jsonl`` from huggingface.co/datasets/THUDM/LVBench, already in this format."""

    records = [json.loads(line) for line in annotations.read_text().splitlines() if line.strip()]
    paths = {record["key"]: videos / f"{record['key']}.mp4" for record in records}
    return records, paths, probe_videos(paths, probe_opencv, workers)


def mlvu_test(annotations: Path, videos: Path, workers: int):
    """``test-ground-truth/test_mcq_gt.json`` from huggingface.co/datasets/MLVU/MLVU_Test."""

    records: dict[str, dict[str, Any]] = {}
    for item in json.loads(annotations.read_text()):
        key = Path(item["video"]).stem
        record = records.setdefault(key, {"key": key, "type": key.split("_")[1], "qa": [], "video_info": None})
        record["qa"].append({
            "uid": item["question_id"],
            "question": _question(item["question"].rstrip(), item["candidates"]),
            "answer": _letter(item["candidates"].index(item["answer"])),
            "question_type": [item["question_type"]],
        })
    paths = {key: videos / f"{key}.mp4" for key in records}
    return list(records.values()), paths, probe_videos(paths, probe_opencv, workers)


def mlvu_dev(annotations: Path, videos: Path, workers: int):
    """``MLVU/json/<task>.json`` from huggingface.co/datasets/MLVU/MVLU, multiple-choice tasks only.

    A file name can appear in several task folders, so videos are keyed ``dev_<task>_<file stem>``.
    """

    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for folder, task in MLVU_DEV_TASKS.items():
        for count, entry in enumerate(json.loads((annotations / f"{folder}.json").read_text())):
            key = f"dev_{task}_{Path(entry['video']).stem}"
            candidates = [" ".join(str(candidate).split()) for candidate in entry["candidates"]]
            record = records.setdefault(key, {"key": key, "type": task, "qa": [], "video_info": None})
            record["qa"].append({
                "uid": f"{task}_{count}",
                "question": _question(" ".join(str(entry["question"]).split()), candidates),
                "answer": _letter(list(entry["candidates"]).index(entry["answer"])),
                "question_type": [task],
            })
            paths[key] = videos / folder / entry["video"]
    paths = {key: paths[key] for key in sorted(records)}
    return [records[key] for key in paths], paths, probe_videos(paths, probe_decord, workers)


def videomme(annotations: Path, videos: Path, workers: int):
    """``videomme/test-00000-of-00001.parquet`` from huggingface.co/datasets/lmms-lab/Video-MME."""

    import pyarrow.parquet as pq

    rows = pq.read_table(annotations).to_pylist()
    paths = {row["videoID"]: videos / f"{row['videoID']}.mp4" for row in rows}
    info = probe_videos(paths, probe_opencv, workers)
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["videoID"]
        if key not in records:
            records[key] = {
                "key": key,
                "type": row["sub_category"],
                "domain": row["domain"],
                "sub_category": row["sub_category"],
                "duration": row["duration"],
                "qa": [],
                "video_info": {
                    "duration_minutes": info[key]["duration_sec"] / 60,
                    "fps": info[key]["fps"],
                    "frames": info[key]["frames"],
                    "resolution": None,
                    "source_video_id": row["video_id"],
                    "url": row["url"],
                },
            }
        # Options come as "A. text".
        options = [option.split(". ", 1)[1] for option in row["options"]]
        records[key]["qa"].append({
            "uid": row["question_id"],
            "question": _question(row["question"], options),
            "answer": row["answer"],
            "question_type": [row["task_type"]],
            "time_reference": None,
            "options": list(row["options"]),
        })
    return list(records.values()), paths, info


def longvideobench(annotations: Path, videos: Path, workers: int):
    """``lvb_val.json`` from huggingface.co/datasets/longvideobench/LongVideoBench."""

    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for item in json.loads(annotations.read_text()):
        key = item["video_id"]
        if key not in records:
            records[key] = {
                "key": key,
                "type": item["topic_category"],
                "domain": "LongVideoBench",
                "sub_category": item["topic_category"],
                "duration": item["duration_group"],
                **{name: item[name] for name in ("video_path", "subtitle_path", "starting_timestamp_for_subtitles", "view_count")},
                "qa": [],
            }
            paths[key] = videos / item["video_path"]
        candidates = [candidate.strip() for candidate in item["candidates"]]
        records[key]["qa"].append({
            "uid": item["id"],
            "question": _question(item["question"], candidates),
            "answer": _letter(item["correct_choice"]),
            "question_type": [item["question_category"], item["level"]],
            "time_reference": None,
            "options": [f"{_letter(i)}. {candidate}" for i, candidate in enumerate(candidates)],
            **{name: item[name] for name in LONGVIDEOBENCH_QA_FIELDS},
        })
    return list(records.values()), paths, probe_videos(paths, probe_ffprobe, workers)


BENCHMARKS = {
    "lvbench": lvbench,
    "mlvu_test": mlvu_test,
    "mlvu_dev": mlvu_dev,
    "videomme": videomme,
    "longvideobench": longvideobench,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--annotations", type=Path, required=True, help="Official annotation file (MLVU-Dev: the MLVU/json directory).")
    parser.add_argument("--videos", type=Path, required=True, help="Directory of the benchmark's videos (MLVU-Dev: MLVU/video).")
    parser.add_argument("--data-root", type=Path, required=True, help="The --data-root of adaalloc.run.")
    parser.add_argument("--workers", type=int, default=8, help="Videos probed in parallel.")
    args = parser.parse_args()

    records, paths, info = BENCHMARKS[args.benchmark](args.annotations, args.videos.resolve(), args.workers)
    out = args.data_root / OUTPUT_DIRS[args.benchmark]
    out.mkdir(parents=True, exist_ok=True)
    (out / "video_qa.json").write_text("".join(json.dumps(record) + "\n" for record in records))
    (out / "video_ids.json").write_text(json.dumps(list(paths), indent=2) + "\n")
    (out / "video_paths.json").write_text(json.dumps({key: str(path) for key, path in paths.items()}, indent=2) + "\n")
    (out / "video_info.json").write_text(json.dumps(info, indent=2) + "\n")
    questions = sum(len(record["qa"]) for record in records)
    print(f"{out}: {len(records)} videos, {questions} questions")


if __name__ == "__main__":
    main()
