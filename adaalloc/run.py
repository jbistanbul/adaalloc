"""Run AdaAlloc on a dataset, in one process or as workers of a shared queue, and merge the results.

Single process (resumes from the ``<output>.jsonl`` ledger)::

    python -m adaalloc.run --config configs/adaalloc_128f.yaml --dataset configs/datasets/lvbench.yaml \
        --data-root DATA --output outputs/lvbench_128f.json

Cluster: every worker runs the same command with ``--work-dir DIR`` (rank and world default to
``$SLURM_PROCID`` / ``$SLURM_NTASKS``) and claims questions from a lock-free queue on the shared
filesystem; ``--merge DIR`` then writes the merged ``--output`` (see ``scripts/slurm_run.sh``).

Next to ``--output`` go the full results (``.json``), one row per answered question (``.json.jsonl``),
the accuracy summary (``.json.metrics.json``) and the config (``.json.config.json``).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable

from tqdm import tqdm

from .config import Config, load_config, load_dataset
from .data import QuestionResult, QuestionSample, load_samples, sample_identity
from .evaluate import build_accuracy_summary

FULL_JSON_FLUSH_EVERY = 50
IDENTITY_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


# ----------------------------------------------------------------------------------------------------
# Result files
# ----------------------------------------------------------------------------------------------------


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def result_to_payload(result: QuestionResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, QuestionResult):
        return result.to_dict()
    return dict(result)


def _payload_sample(payload: dict[str, Any]) -> dict[str, Any]:
    sample = payload.get("sample")
    return sample if isinstance(sample, dict) else payload


def result_identity(payload: dict[str, Any]) -> str:
    """Identity of a full result payload or of a ledger row."""

    return sample_identity(_payload_sample(payload))


def _sibling_path(output_path: str | Path, suffix: str) -> Path:
    output = Path(output_path)
    return output.with_name(output.name + suffix)


def ledger_path_for(output_path: str | Path) -> Path:
    return _sibling_path(output_path, ".jsonl")


def metrics_path_for(output_path: str | Path) -> Path:
    return _sibling_path(output_path, ".metrics.json")


def config_path_for(output_path: str | Path) -> Path:
    return _sibling_path(output_path, ".config.json")


def build_config_snapshot(config: Config, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_config = asdict(config)
    resolved_config["budget"].update(visual_tokens=config.budget.visual_tokens, total_pixels=config.budget.total_pixels)
    canonical_config = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(canonical_config.encode("utf-8")).hexdigest(),
        "config": resolved_config,
        "metadata": dict(metadata or {}),
    }


def write_config_snapshot(path: str | Path, config: Config, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot = build_config_snapshot(config, metadata=metadata)
    write_json_atomic(Path(path), snapshot)
    return snapshot


def minimal_ledger_payload(result: QuestionResult | dict[str, Any]) -> dict[str, Any]:
    """The ledger row of a result: identity, gold letter, prediction and frame/token counts."""

    payload = result_to_payload(result)
    if "sample" not in payload:
        return payload
    sample = payload["sample"]
    prediction = payload["prediction"]
    selected_frames = payload.get("selected_frames")
    qwen_cache = (payload.get("metadata") or {}).get("qwen_cache") or {}
    return {
        "dataset_name": sample["dataset_name"],
        "video_id": sample["video_id"],
        "question_uid": sample["question_uid"],
        "question_text": sample["question_text"],
        "answer_letter": sample["answer_letter"],
        "selected_option": prediction["selected_option"],
        "confidence": prediction["confidence"],
        "reasoning": prediction["reasoning"],
        "selected_option_text": prediction.get("selected_option_text"),
        "selected_frame_count": (
            len(selected_frames) if isinstance(selected_frames, list) else payload.get("selected_frame_count")
        ),
        "decoded_frame_count": qwen_cache.get("decoded_frame_count", payload.get("decoded_frame_count")),
        "visual_token_count": qwen_cache.get("visual_token_count", payload.get("visual_token_count")),
    }


def _payload_from_ledger_row(row: dict[str, Any], sample_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a result payload from a ledger row whose full result was never flushed."""

    if "sample" in row:
        return row
    sample = sample_lookup.get(result_identity(row))
    if sample is None:
        return row
    return {
        "sample": sample,
        "prediction": {key: row.get(key) for key in ("selected_option", "confidence", "reasoning", "selected_option_text")},
        "selected_frame_count": row.get("selected_frame_count"),
        "decoded_frame_count": row.get("decoded_frame_count"),
        "visual_token_count": row.get("visual_token_count"),
    }


def load_result_ledger(path: str | Path) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    with ledger_path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_result_ledger(path: str | Path, result: QuestionResult | dict[str, Any]) -> dict[str, Any]:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = minimal_ledger_payload(result)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return payload


def load_results_json(path: str | Path) -> list[dict[str, Any]]:
    output_path = Path(path)
    if not output_path.exists():
        return []
    loaded = json.loads(output_path.read_text())
    if not isinstance(loaded, list):
        raise TypeError(f"Expected list payload in {output_path}")
    return [dict(item) for item in loaded]


def write_results(path: str | Path, results: list[QuestionResult | dict[str, Any]]) -> None:
    write_json_atomic(Path(path), [result_to_payload(result) for result in results])


def _report_and_save_accuracy(
    config: Config,
    output_path: str | Path,
    results: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the config snapshot and accuracy summary next to ``output_path`` and print the accuracy."""

    output = Path(output_path).resolve()
    paths = {
        "predictions_path": str(output),
        "ledger_path": str(ledger_path_for(output)),
        "metrics_path": str(metrics_path_for(output)),
    }
    config_snapshot_path = config_path_for(output)
    config_snapshot = write_config_snapshot(config_snapshot_path, config, metadata=paths)
    summary = build_accuracy_summary(
        results,
        metadata={
            "experiment_name": config.name,
            "dataset_name": config.dataset.name,
            "visual_tokens": config.budget.visual_tokens,
            "predictions_path": paths["predictions_path"],
            "ledger_path": paths["ledger_path"],
            "config_path": str(config_snapshot_path),
            "config_sha256": config_snapshot["config_sha256"],
            **(metadata or {}),
        },
    )
    summary_path = metrics_path_for(output)
    write_json_atomic(summary_path, summary)
    print(f"Final accuracy: {summary['correct']}/{summary['num_samples']} ({summary['accuracy']:.2%})")
    if summary.get("avg_selected_frames") is not None:
        print(f"Average selected frames: {summary['avg_selected_frames']:.2f}")
    if summary.get("avg_decoded_frame_count") is not None:
        print(f"Average decoded frames: {summary['avg_decoded_frame_count']:.2f}")
    print(f"Saved metrics summary to {summary_path}")
    return summary


# ----------------------------------------------------------------------------------------------------
# Distributed work queue
# ----------------------------------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class WorkItem:
    index: int
    identity: str


def safe_identity(identity: str) -> str:
    safe = IDENTITY_SAFE_PATTERN.sub("_", identity).strip("._") or "item"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:120]}-{digest}"


class DistributedWorkQueue:
    """Shared-filesystem manifest and claim queue for Slurm workers.

    The queue uses only atomic directory creation / atomic rename, so it runs on cluster scratch
    filesystems without a service.
    """

    def __init__(self, work_dir: str | Path, *, ready_timeout_sec: float = 300.0) -> None:
        self.work_dir = Path(work_dir)
        self.items_path = self.work_dir / "items.jsonl"
        self.ready_path = self.work_dir / "manifest.ready"
        self.lock_dir = self.work_dir / "manifest.lock"
        self.claims_dir = self.work_dir / "claims"
        self.done_dir = self.work_dir / "done"
        self.workers_dir = self.work_dir / "workers"
        self.failures_dir = self.work_dir / "failures"
        self.ready_timeout_sec = ready_timeout_sec

    def ensure_manifest(self, identities: Iterable[str], *, rank: int = 0, run_id: str | None = None) -> list[WorkItem]:
        """Write the manifest once (the first rank to take the lock) and return it to every rank."""

        for directory in (self.work_dir, self.claims_dir, self.done_dir, self.workers_dir, self.failures_dir):
            directory.mkdir(parents=True, exist_ok=True)

        if self.ready_path.exists():
            return self.load_manifest()

        try:
            self.lock_dir.mkdir()
            acquired = True
        except FileExistsError:
            acquired = False

        if acquired:
            items = [WorkItem(index=index, identity=identity) for index, identity in enumerate(identities)]
            tmp_path = self.items_path.with_name(self.items_path.name + f".tmp-{os.getpid()}")
            with tmp_path.open("w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps({"index": item.index, "identity": item.identity}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.items_path)
            write_json_atomic(
                self.ready_path,
                {
                    "run_id": run_id,
                    "created_by_rank": rank,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "num_items": len(items),
                },
            )
            return items

        deadline = time.monotonic() + self.ready_timeout_sec
        while time.monotonic() < deadline:
            if self.ready_path.exists():
                return self.load_manifest()
            time.sleep(0.2)
        raise TimeoutError(f"Timed out waiting for distributed manifest readiness at {self.ready_path}")

    def load_manifest(self) -> list[WorkItem]:
        if not self.ready_path.exists():
            raise RuntimeError(f"Distributed manifest is not ready: {self.ready_path}")
        if not self.items_path.exists():
            raise RuntimeError(f"Distributed manifest items file is missing: {self.items_path}")
        items: list[WorkItem] = []
        with self.items_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if "identity" not in payload or "index" not in payload:
                    raise RuntimeError(f"Malformed manifest item at {self.items_path}:{line_no}")
                items.append(WorkItem(index=int(payload["index"]), identity=str(payload["identity"])))
        return items

    def is_done(self, identity: str) -> bool:
        return self.done_path(identity).exists()

    def claim_path(self, identity: str) -> Path:
        return self.claims_dir / f"{safe_identity(identity)}.claim"

    def done_path(self, identity: str) -> Path:
        return self.done_dir / f"{safe_identity(identity)}.done"

    def claim_next(
        self,
        items: Iterable[WorkItem],
        *,
        worker_rank: int,
        worker_id: str | None = None,
        claim_timeout_sec: float | None = None,
    ) -> WorkItem | None:
        """Claim the first item that is neither done nor claimed; claims older than the timeout are stale."""

        now = time.time()
        for item in items:
            if self.is_done(item.identity):
                continue
            claim_dir = self.claim_path(item.identity)
            if claim_dir.exists() and claim_timeout_sec is not None:
                metadata_path = claim_dir / "metadata.json"
                try:
                    metadata = json.loads(metadata_path.read_text())
                    claimed_at = float(metadata.get("claimed_at_epoch", 0.0))
                except Exception:
                    claimed_at = 0.0
                if claimed_at and now - claimed_at > claim_timeout_sec and not self.is_done(item.identity):
                    try:
                        metadata_path.unlink(missing_ok=True)
                        claim_dir.rmdir()
                    except OSError:
                        pass
            try:
                claim_dir.mkdir(parents=True)
            except FileExistsError:
                continue
            write_json_atomic(
                claim_dir / "metadata.json",
                {
                    "identity": item.identity,
                    "index": item.index,
                    "worker_rank": worker_rank,
                    "worker_id": worker_id or f"worker-{worker_rank}",
                    "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "claimed_at_epoch": now,
                    "pid": os.getpid(),
                },
            )
            return item
        return None

    def mark_done(self, identity: str, *, worker_rank: int) -> None:
        write_json_atomic(
            self.done_path(identity),
            {
                "identity": identity,
                "worker_rank": worker_rank,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

    def worker_output_path(self, worker_rank: int) -> Path:
        return self.workers_dir / f"worker-{worker_rank}.json"

    def worker_ledger_path(self, worker_rank: int) -> Path:
        return self.workers_dir / f"worker-{worker_rank}.jsonl"

    def failure_path(self, worker_rank: int) -> Path:
        return self.failures_dir / f"worker-{worker_rank}.json"


# ----------------------------------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------------------------------


def _ordered_runtime_samples(config: Config, max_samples: int | None = None) -> list[QuestionSample]:
    samples = load_samples(config.dataset)
    return samples if max_samples is None else samples[:max_samples]


def _build_answerer(config: Config) -> Callable[[QuestionSample], QuestionResult]:
    """Load UniversalVTG and Qwen once and return ``answer(sample) -> QuestionResult``."""

    # Imported here so merging and scoring need no GPU stack.
    from .pipeline import answer_question_adaalloc, prepare_sample
    from .qwen import build_reasoner
    from .retrieval import UniversalVTGBackend

    retrieval_backend = UniversalVTGBackend(config)
    reasoner = build_reasoner(config)

    def answer(sample: QuestionSample) -> QuestionResult:
        return answer_question_adaalloc(config, prepare_sample(config, sample), retrieval_backend, reasoner)

    return answer


def run_experiment(config: Config, output_path: str | Path, *, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Answer every question not yet in the ledger next to ``output_path``, then write outputs and metrics."""

    samples = _ordered_runtime_samples(config, max_samples)
    sample_order = {sample_identity(sample): index for index, sample in enumerate(samples)}
    sample_lookup = {sample_identity(sample): sample.to_dict() for sample in samples}

    # Resume: flushed full results, plus ledger rows answered after the last flush.
    ledger_path = ledger_path_for(output_path)
    results = {result_identity(payload): payload for payload in load_results_json(output_path)}
    for row in load_result_ledger(ledger_path):
        results.setdefault(result_identity(row), _payload_from_ledger_row(row, sample_lookup))
    pending_samples = [sample for sample in samples if sample_identity(sample) not in results]

    def _ordered_result_payloads() -> list[dict[str, Any]]:
        return sorted(results.values(), key=lambda payload: sample_order.get(result_identity(payload), len(sample_order)))

    if pending_samples:
        answer = _build_answerer(config)
        completed_since_flush = 0
        for sample in tqdm(pending_samples, desc="AdaAlloc", unit="question"):
            result = answer(sample)
            append_result_ledger(ledger_path, result)
            payload = result.to_dict()
            results[result_identity(payload)] = payload
            completed_since_flush += 1
            if completed_since_flush >= FULL_JSON_FLUSH_EVERY:
                write_results(output_path, _ordered_result_payloads())
                completed_since_flush = 0

    ordered_results = _ordered_result_payloads()
    write_results(output_path, ordered_results)
    _report_and_save_accuracy(config, output_path, ordered_results, metadata={"max_samples": max_samples})
    return ordered_results


def run_dynamic_work(
    config: Config,
    *,
    work_dir: str | Path,
    worker_rank: int,
    worker_count: int,
    max_samples: int | None = None,
    claim_timeout_sec: float | None = None,
) -> list[QuestionResult]:
    """Run one long-lived worker that claims questions from a shared work queue until none is left."""

    if not 0 <= worker_rank < worker_count:
        raise ValueError(f"Invalid worker rank/world: rank={worker_rank}, world={worker_count}")
    queue = DistributedWorkQueue(work_dir)
    samples = _ordered_runtime_samples(config, max_samples)
    items = queue.ensure_manifest(
        [sample_identity(sample) for sample in samples],
        rank=worker_rank,
        run_id=os.environ.get("SLURM_JOB_ID"),
    )
    sample_lookup = {sample_identity(sample): sample for sample in samples}
    print(f"[worker {worker_rank}/{worker_count}] {len(items)} questions in {Path(work_dir).resolve()}", flush=True)

    worker_output_path = queue.worker_output_path(worker_rank)
    worker_ledger_path = queue.worker_ledger_path(worker_rank)
    answer = _build_answerer(config)
    results: list[dict[str, Any]] = load_results_json(worker_output_path)
    completed_full_ids = {result_identity(payload) for payload in results}
    completed_since_flush = 0
    returned_results: list[QuestionResult] = []

    try:
        while True:
            item = queue.claim_next(
                items,
                worker_rank=worker_rank,
                worker_id=f"worker-{worker_rank}",
                claim_timeout_sec=claim_timeout_sec,
            )
            if item is None:
                break
            sample = sample_lookup.get(item.identity)
            if sample is None:
                raise RuntimeError(f"Manifest identity is not present in loaded samples: {item.identity}")
            if item.identity in completed_full_ids:
                queue.mark_done(item.identity, worker_rank=worker_rank)
                continue
            result = answer(sample)
            append_result_ledger(worker_ledger_path, result)
            payload = result.to_dict()
            identity = result_identity(payload)
            if identity not in completed_full_ids:
                results.append(payload)
                completed_full_ids.add(identity)
            returned_results.append(result)
            completed_since_flush += 1
            if completed_since_flush >= FULL_JSON_FLUSH_EVERY:
                write_results(worker_output_path, results)
                completed_since_flush = 0
            queue.mark_done(item.identity, worker_rank=worker_rank)
    except Exception as exc:
        write_json_atomic(
            queue.failure_path(worker_rank),
            {"worker_rank": worker_rank, "error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        if completed_since_flush or not worker_output_path.exists():
            write_results(worker_output_path, results)

    print(f"[worker {worker_rank}/{worker_count}] answered {len(returned_results)} questions", flush=True)
    return returned_results


def _write_merged_payloads(
    config: Config,
    output_path: str | Path,
    payloads: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    write_results(output_path, payloads)
    ledger_path = ledger_path_for(output_path)
    ledger_path.write_text("".join(json.dumps(minimal_ledger_payload(payload)) + "\n" for payload in payloads))
    return _report_and_save_accuracy(config, output_path, payloads, metadata=metadata)


def _merge_work_dir_outputs(config: Config, work_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Merge a finished work directory into ``output_path``.

    Every manifest question must be done and answered; a question answered by several workers (a
    retried claim) counts once when the answers agree and is rejected when they differ.
    """

    queue = DistributedWorkQueue(work_dir)
    expected_ids = [item.identity for item in queue.load_manifest()]
    expected_id_set = set(expected_ids)

    # One payload per (worker, question): the flushed full result, else the ledger row.
    per_worker: dict[tuple[str, str], dict[str, Any]] = {}
    for worker_json in sorted(queue.workers_dir.glob("worker-*.json")):
        for payload in load_results_json(worker_json):
            per_worker[(worker_json.stem, result_identity(payload))] = payload
    for worker_ledger in sorted(queue.workers_dir.glob("worker-*.jsonl")):
        for row in load_result_ledger(worker_ledger):
            key = (worker_ledger.stem, result_identity(row))
            if "sample" not in per_worker.get(key, {}):
                per_worker[key] = row

    by_identity: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for (worker_name, identity), payload in per_worker.items():
        if identity in expected_id_set:
            by_identity.setdefault(identity, []).append((worker_name, payload))

    missing_results = [identity for identity in expected_ids if identity not in by_identity]
    missing_done = [identity for identity in expected_ids if not queue.is_done(identity)]
    conflicts: dict[str, list[str]] = {}
    identical_resume_duplicates = 0
    for identity, worker_payloads in by_identity.items():
        if len(worker_payloads) <= 1:
            continue
        # A replacement worker may redo a question whose earlier claim finished just before a node
        # failure: count an identical answer once, reject a different one.
        first_row = minimal_ledger_payload(worker_payloads[0][1])
        if all(minimal_ledger_payload(payload) == first_row for _worker, payload in worker_payloads[1:]):
            identical_resume_duplicates += 1
        else:
            conflicts[identity] = sorted(worker_name for worker_name, _payload in worker_payloads)
    if missing_results or missing_done or conflicts:
        parts: list[str] = []
        if missing_results:
            parts.append(f"{len(missing_results)} missing result(s): {missing_results[:10]}")
        if missing_done:
            parts.append(f"{len(missing_done)} missing done marker(s): {missing_done[:10]}")
        if conflicts:
            parts.append(f"conflicting worker completion(s): {dict(list(conflicts.items())[:10])}")
        raise RuntimeError("Distributed work directory is incomplete or conflicting: " + "; ".join(parts))

    sample_lookup = {sample_identity(sample): sample.to_dict() for sample in load_samples(config.dataset)}
    merged_payloads = []
    for identity in expected_ids:
        payloads = [payload for _worker, payload in by_identity[identity]]
        best_payload = next((payload for payload in payloads if "sample" in payload), payloads[0])
        merged_payloads.append(_payload_from_ledger_row(best_payload, sample_lookup))

    return _write_merged_payloads(
        config,
        output_path,
        merged_payloads,
        metadata={
            "distributed_evaluation": True,
            "work_dir": str(Path(work_dir).resolve()),
            "identical_resume_duplicate_count": identical_resume_duplicates,
        },
    )


# ----------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adaalloc.run",
        description="Answer a multiple-choice video QA dataset with AdaAlloc and score it.",
    )
    parser.add_argument("--config", required=True, help="Method config, e.g. configs/adaalloc_128f.yaml (32K) or configs/adaalloc_64f.yaml (16K).")
    parser.add_argument("--dataset", required=True, help="Dataset config, e.g. configs/datasets/lvbench.yaml.")
    parser.add_argument("--data-root", default=None, help="Root of relative dataset paths (default: the dataset config's directory).")
    parser.add_argument("--output", required=True, help="Predictions .json; ledger, metrics and config snapshot go next to it.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only answer the first N questions.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--work-dir", default=None, help="Run as one worker of the shared queue in this directory.")
    mode.add_argument("--merge", metavar="WORK_DIR", default=None, help="Merge a finished work directory into --output.")
    parser.add_argument("--rank", type=int, default=None, help="Worker rank (default: $SLURM_PROCID).")
    parser.add_argument("--world", type=int, default=None, help="Number of workers (default: $SLURM_NTASKS).")
    parser.add_argument(
        "--claim-timeout-sec",
        type=float,
        default=None,
        help="Reclaim questions claimed longer ago than this (by workers that died); must exceed the slowest question.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.work_dir is None and (args.rank is not None or args.world is not None or args.claim_timeout_sec is not None):
        parser.error("--rank, --world and --claim-timeout-sec require --work-dir")

    config = load_config(args.config)
    config.dataset = load_dataset(args.dataset, args.data_root)

    if args.merge is not None:
        _merge_work_dir_outputs(config, args.merge, args.output)
    elif args.work_dir is not None:
        run_dynamic_work(
            config,
            work_dir=args.work_dir,
            worker_rank=args.rank if args.rank is not None else int(os.environ.get("SLURM_PROCID", "0")),
            worker_count=args.world if args.world is not None else int(os.environ.get("SLURM_NTASKS", "1")),
            max_samples=args.max_samples,
            claim_timeout_sec=args.claim_timeout_sec,
        )
    else:
        run_experiment(config, args.output, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
