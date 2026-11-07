"""Score predictions: pooled accuracy and, for MLVU, M-Avg (the mean of per-task accuracies).

    python -m adaalloc.evaluate outputs/lvbench_32k.json
    python -m adaalloc.evaluate outputs/mlvu_32k.json --mavg [--annotations DATA/MLVU/annotations/video_qa.json]

Accepts the full ``.json`` output or its ``.json.jsonl`` ledger. A question is correct when the stripped
predicted letter equals the gold letter; a missing prediction counts as wrong. MLVU is reported as
M-Avg, which differs from the pooled accuracy whenever task sizes differ. A question's task is
``question_type[0]``: from ``--annotations`` (every annotated question then counts, unanswered ones as
wrong), else from the results themselves (full ``.json`` output only).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import _read_records


def extract_gold_and_pred(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    sample = payload.get("sample", {})
    prediction = payload.get("prediction", {})
    gold = payload.get("answer_letter", sample.get("answer_letter"))
    pred = payload.get("selected_option", prediction.get("selected_option"))
    gold_text = str(gold).strip() if gold is not None else None
    pred_text = str(pred).strip() if pred is not None else None
    return gold_text or None, pred_text or None


def _is_correct(payload: dict[str, Any]) -> bool:
    gold, pred = extract_gold_and_pred(payload)
    return gold is not None and gold == pred


def build_accuracy_summary(results: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pooled accuracy plus mean confidence and frame/token counts over full results or ledger rows."""

    total = len(results)
    correct = 0
    missing = 0
    confidence_total = 0.0
    confidence_count = 0
    selected_frame_total = 0
    selected_frame_count = 0
    decoded_frame_total = 0.0
    decoded_frame_count = 0
    visual_token_total = 0.0
    visual_token_count = 0

    for result in results:
        gold, pred = extract_gold_and_pred(result)
        if gold is None or pred is None:
            missing += 1
        else:
            correct += gold == pred

        prediction = result.get("prediction", {})
        confidence = result.get("confidence", prediction.get("confidence"))
        if confidence is not None:
            confidence_total += float(confidence)
            confidence_count += 1

        selected_frames = result.get("selected_frames")
        if isinstance(selected_frames, list):
            selected_frame_total += len(selected_frames)
            selected_frame_count += 1
        else:
            selected_frame_count_value = result.get("selected_frame_count")
            if isinstance(selected_frame_count_value, (int, float)):
                selected_frame_total += int(selected_frame_count_value)
                selected_frame_count += 1

        qwen_cache = (result.get("metadata") or {}).get("qwen_cache") or {}
        decoded = qwen_cache.get("decoded_frame_count")
        if not isinstance(decoded, (int, float)):
            decoded = result.get("decoded_frame_count")
        if isinstance(decoded, (int, float)):
            decoded_frame_total += float(decoded)
            decoded_frame_count += 1

        visual_tokens = qwen_cache.get("visual_token_count")
        if not isinstance(visual_tokens, (int, float)):
            visual_tokens = result.get("visual_token_count")
        if isinstance(visual_tokens, (int, float)):
            visual_token_total += float(visual_tokens)
            visual_token_count += 1

    summary = {
        "num_samples": total,
        "correct": correct,
        "missing_predictions": missing,
        "accuracy": (correct / total) if total else 0.0,
        "mean_confidence": (confidence_total / confidence_count) if confidence_count else None,
        "avg_selected_frames": (selected_frame_total / selected_frame_count) if selected_frame_count else None,
        "avg_decoded_frame_count": (decoded_frame_total / decoded_frame_count) if decoded_frame_count else None,
        "avg_visual_token_count": (visual_token_total / visual_token_count) if visual_token_count else None,
    }
    if metadata:
        summary.update(metadata)
    return summary


def _question_key(payload: dict[str, Any]) -> tuple[str, str]:
    sample = payload.get("sample")
    sample = sample if isinstance(sample, dict) else payload
    return sample["video_id"], str(sample["question_uid"])


def task_by_question(annotations: str | Path) -> dict[tuple[str, str], str]:
    """Map ``(video_id, question_uid)`` to the question's task in an annotations file."""

    tasks: dict[tuple[str, str], str] = {}
    for record in _read_records(annotations):
        for qa in record["qa"]:
            task = qa.get("question_type") or [record.get("type", "unknown")]
            tasks[(record["key"], str(qa["uid"]))] = task[0] if isinstance(task, list) else task
    return tasks


def mavg_summary(results: list[dict[str, Any]], annotations: str | Path | None = None) -> dict[str, Any]:
    """Per-task accuracies, their mean (M-Avg) and the pooled accuracy over the same questions."""

    by_question = {_question_key(payload): payload for payload in results}
    if annotations is not None:
        tasks = task_by_question(annotations)
    else:
        tasks = {}
        for key, payload in by_question.items():
            sample = payload.get("sample")
            if not isinstance(sample, dict):
                raise ValueError("Ledger rows carry no task: pass --annotations or score the full .json output.")
            tasks[key] = (sample.get("question_types") or ["unknown"])[0]

    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for key, task in tasks.items():
        total[task] += 1
        payload = by_question.get(key)
        if payload is not None and _is_correct(payload):
            correct[task] += 1
    per_task = {
        task: {"correct": correct[task], "total": total[task], "accuracy": correct[task] / total[task]}
        for task in sorted(total)
    }
    num_questions = sum(total.values())
    return {
        "tasks": per_task,
        "m_avg": sum(entry["accuracy"] for entry in per_task.values()) / len(per_task) if per_task else 0.0,
        "pooled_accuracy": sum(correct.values()) / num_questions if num_questions else 0.0,
        "correct": sum(correct.values()),
        "num_questions": num_questions,
        "missing_predictions": sum(1 for key in tasks if key not in by_question),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m adaalloc.evaluate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, help="Predictions .json or its .json.jsonl ledger.")
    parser.add_argument("--mavg", action="store_true", help="Also report MLVU M-Avg (mean of per-task accuracies).")
    parser.add_argument("--annotations", type=Path, default=None, help="Annotations file giving each question's task (with --mavg).")
    args = parser.parse_args(argv)
    if args.annotations is not None and not args.mavg:
        parser.error("--annotations is only used with --mavg")

    results = _read_records(args.predictions)
    if args.mavg and args.annotations is None and any(not isinstance(result.get("sample"), dict) for result in results):
        parser.error("ledger rows carry no task: pass --annotations or score the full .json output")
    summary = build_accuracy_summary(results)
    print(f"Accuracy: {summary['correct']}/{summary['num_samples']} ({100 * summary['accuracy']:.2f}%)")
    if summary["missing_predictions"]:
        print(f"WARNING: {summary['missing_predictions']} results have no prediction and count as wrong.")
    if not args.mavg:
        return

    mavg = mavg_summary(results, args.annotations)
    print(f"\n{'task':16s} {'correct':>8s} {'total':>6s} {'acc':>7s}")
    for task, entry in mavg["tasks"].items():
        print(f"{task:16s} {entry['correct']:8d} {entry['total']:6d} {100 * entry['accuracy']:6.2f}%")
    print(f"\nM-Avg (mean of {len(mavg['tasks'])} task accuracies): {100 * mavg['m_avg']:.2f}%")
    print(f"Pooled accuracy: {100 * mavg['pooled_accuracy']:.2f}% ({mavg['correct']}/{mavg['num_questions']})")
    if mavg["missing_predictions"]:
        print(f"WARNING: {mavg['missing_predictions']} annotated questions have no prediction and count as wrong.")


if __name__ == "__main__":
    main()
