#!/bin/bash
#SBATCH --job-name=adaalloc
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1       # workers per node: one per GPU
#SBATCH --gpus-per-node=1         # = --ntasks-per-node (drop if your cluster selects GPU nodes by partition)
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/adaalloc_%j.out

# Runs AdaAlloc on one dataset with (nodes x workers per node) workers, one GPU each. Workers claim
# questions from a shared work directory until none is left; the results are then merged into OUTPUT
# (plus OUTPUT.jsonl, OUTPUT.metrics.json and OUTPUT.config.json). Submit from the repository root:
#
#   mkdir -p slurm_logs
#   CONFIG=configs/adaalloc_128f.yaml DATASET=configs/datasets/lvbench.yaml DATA_ROOT=/path/to/data \
#   OUTPUT=outputs/lvbench_128f.json sbatch scripts/slurm_run.sh
#
# Optional environment:
#   MAX_SAMPLES             only the first N questions
#   WORK_DIR                shared work directory (default: $OUTPUT.work/run-$SLURM_JOB_ID)
#   CLAIM_TIMEOUT_SEC       reclaim questions claimed longer ago than this by workers that died; must
#                           exceed the slowest question (e.g. 3600). To finish an interrupted run, resubmit
#                           with WORK_DIR=<its work dir> and CLAIM_TIMEOUT_SEC set.
#   PYTHON                  interpreter (default: python)
#   SKIP_PREFLIGHT=1        skip the input checks
#   MERGE_RETRIES           merge attempts after the workers finish (default 5)
#   MERGE_RETRY_SLEEP_SEC   seconds between merge attempts (default 30; lets the shared filesystem settle)

set -euo pipefail

CONFIG="${CONFIG:?Set CONFIG, e.g. configs/adaalloc_128f.yaml}"
DATASET="${DATASET:?Set DATASET, e.g. configs/datasets/lvbench.yaml}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the directory holding the datasets}"
OUTPUT="${OUTPUT:?Set OUTPUT, e.g. outputs/lvbench_128f.json}"
WORK_DIR="${WORK_DIR:-$OUTPUT.work/run-${SLURM_JOB_ID:-manual}}"
PYTHON="${PYTHON:-python}"
MERGE_RETRIES="${MERGE_RETRIES:-5}"
MERGE_RETRY_SLEEP_SEC="${MERGE_RETRY_SLEEP_SEC:-30}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$(dirname "$OUTPUT")"

# Activate the environment, e.g.
#   source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate adaalloc
# and point to the UniversalVTG checkout if it is not the third_party/universalvtg submodule:
#   export ADAALLOC_UNIVERSAL_VTG_ROOT=/path/to/universalvtg

# On some systems decord's FFmpeg libraries need a newer libstdc++ than the one a loaded compiler
# module puts first: decord then fails to load, and threaded workers abort inside ld.so (a dl-tls.c
# assertion, exit code 127). Preloading the conda environment's own libstdc++ fixes it:
# export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"  # qwen-vl-utils video reader of the paper runs
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-20480}"            # tolerate slow seeks near the end of long videos
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

COMMON_ARGS=(--config "$CONFIG" --dataset "$DATASET" --data-root "$DATA_ROOT" --output "$OUTPUT")
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  COMMON_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  # Check that the dataset files, the PE-Core features and the UniversalVTG checkpoint exist.
  CONFIG="$CONFIG" DATASET="$DATASET" DATA_ROOT="$DATA_ROOT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

from adaalloc.config import load_config, load_dataset
from adaalloc.retrieval import _resolve_universal_vtg_root

config = load_config(os.environ["CONFIG"])
dataset = load_dataset(os.environ["DATASET"], os.environ["DATA_ROOT"])
required = {key: getattr(dataset, key) for key in ("annotations", "video_ids", "video_paths", "video_info", "pecore_features")}
root = _resolve_universal_vtg_root(config.retrieval.universal_vtg_root)
required["UniversalVTG checkpoint"] = root / config.retrieval.experiment / "models" / f"{config.retrieval.checkpoint}.pth"
missing = [f"  {name}: {path}" for name, path in required.items() if path is not None and not Path(path).exists()]
if missing:
    raise SystemExit("Missing AdaAlloc inputs:\n" + "\n".join(missing))
print(f"Preflight OK: {dataset.name}, {config.name}")
PY
fi

WORKER_CMD=("$PYTHON" -u -m adaalloc.run "${COMMON_ARGS[@]}" --work-dir "$WORK_DIR")
if [[ -n "${CLAIM_TIMEOUT_SEC:-}" ]]; then
  WORKER_CMD+=(--claim-timeout-sec "$CLAIM_TIMEOUT_SEC")
fi

printf 'Job %s on %s: %s workers\n' "${SLURM_JOB_ID:-<none>}" "$(hostname)" "${SLURM_NTASKS:-1}"
printf 'Work dir: %s\nOutput:   %s\nWorker command:' "$WORK_DIR" "$OUTPUT"
printf ' %q' "${WORKER_CMD[@]}"
printf '\n\n'

# One worker per task; bind each to one GPU of its node (rank and world come from SLURM_PROCID /
# SLURM_NTASKS). If Slurm already gives each task a single GPU, that GPU is used.
BIND_GPU_SCRIPT=$(cat <<'BASH'
set -euo pipefail
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES="${GPUS[$(( ${SLURM_LOCALID:-0} % ${#GPUS[@]} ))]}"
echo "[gpu-bind] host=$(hostname) rank=${SLURM_PROCID:-0} local=${SLURM_LOCALID:-0} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
exec "$@"
BASH
)
srun bash -c "$BIND_GPU_SCRIPT" bash "${WORKER_CMD[@]}"

MERGE_CMD=("$PYTHON" -u -m adaalloc.run "${COMMON_ARGS[@]}" --merge "$WORK_DIR")
for attempt in $(seq 1 "$MERGE_RETRIES"); do
  echo "Merging $WORK_DIR (attempt $attempt/$MERGE_RETRIES)"
  if "${MERGE_CMD[@]}"; then
    exit 0
  fi
  if [[ "$attempt" -lt "$MERGE_RETRIES" ]]; then
    sleep "$MERGE_RETRY_SLEEP_SEC"
  fi
done
echo "Merge failed after $MERGE_RETRIES attempts." >&2
exit 1
