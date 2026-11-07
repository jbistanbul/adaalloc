#!/bin/bash
# Extracts the PE-Core video features that AdaAlloc's UniversalVTG retrieval reads, for every video of
# a dataset, with UniversalVTG's own extractor (PE-Core-L14-336 at 2 fps) on all visible GPUs. Run it
# in the UniversalVTG environment after preparing the dataset's annotations (scripts/prepare_data.py):
#
#   scripts/extract_features.sh configs/datasets/lvbench.yaml /path/to/data
#
# Features go to the dataset's pecore_features directory as <video_id>.pt; videos that already have
# one are skipped, so an interrupted run can simply be restarted. UniversalVTG is found as in
# adaalloc.retrieval: $ADAALLOC_UNIVERSAL_VTG_ROOT, else the third_party/universalvtg submodule. The
# PE-Core-L14-336 weights are downloaded from the Hugging Face Hub (facebook/PE-Core-L14-336) on first use.
#
# Optional environment: PYTHON (default python), BATCH_SIZE (64), WORKERS_PER_GPU (4),
# VIDEO_TIMEOUT_SEC (600, per video).

set -euo pipefail

DATASET="${1:?Usage: $0 DATASET_CONFIG DATA_ROOT}"
DATA_ROOT="${2:?Usage: $0 DATASET_CONFIG DATA_ROOT}"
PYTHON="${PYTHON:-python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIVERSAL_VTG_ROOT="${ADAALLOC_UNIVERSAL_VTG_ROOT:-$REPO/third_party/universalvtg}"
EXTRACTOR="$UNIVERSAL_VTG_ROOT/feature_extraction/extract_visual_features.py"
[[ -f "$EXTRACTOR" ]] || { echo "UniversalVTG extractor not found: $EXTRACTOR" >&2; exit 1; }

# The extractor reads <videos_root>/<video_id>.mp4, so link every video of the dataset under that name.
VIDEOS="$(mktemp -d)"
trap 'rm -rf "$VIDEOS"' EXIT
FEATURES="$(PYTHONPATH="$REPO" "$PYTHON" - "$DATASET" "$DATA_ROOT" "$VIDEOS" <<'PY'
import json
import sys
from pathlib import Path

from adaalloc.config import load_dataset

config_path, data_root, videos = sys.argv[1:]
dataset = load_dataset(config_path, data_root)
video_ids = json.loads(Path(dataset.video_ids).read_text())
video_paths = json.loads(Path(dataset.video_paths).read_text())
missing = [video_id for video_id in video_ids if not Path(video_paths.get(video_id, "")).is_file()]
if missing:
    sys.exit(f"{len(missing)} videos are missing, e.g. {missing[:5]}")
for video_id in video_ids:
    (Path(videos) / f"{video_id}.mp4").symlink_to(Path(video_paths[video_id]).resolve())
# Named after the dataset config: the extractor caches video metadata under this file's name.
(Path(videos) / f"{Path(config_path).stem}.json").write_text(json.dumps(video_ids))
Path(dataset.pecore_features).mkdir(parents=True, exist_ok=True)
print(dataset.pecore_features)
PY
)"

"$PYTHON" "$EXTRACTOR" \
  --videos_root "$VIDEOS" \
  --video_format mp4 \
  --video_uids_json "$VIDEOS/$(basename "${DATASET%.*}").json" \
  --save_dir "$FEATURES" \
  --encoder pe \
  --model_config PE-Core-L14-336 \
  --read_fps 2.0 \
  --batch_size "${BATCH_SIZE:-64}" \
  --dataloader_num_workers "${WORKERS_PER_GPU:-4}" \
  --timeout "${VIDEO_TIMEOUT_SEC:-600}"

# Every video needs a (T, 1024) feature tensor.
"$PYTHON" - "$VIDEOS/$(basename "${DATASET%.*}").json" "$FEATURES" <<'PY'
import json
import sys
from pathlib import Path

import torch

video_ids = json.loads(Path(sys.argv[1]).read_text())
features = Path(sys.argv[2])
bad = []
for video_id in video_ids:
    path = features / f"{video_id}.pt"
    if not path.is_file():
        bad.append(f"{video_id}: missing")
        continue
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.ndim != 2 or value.shape[-1] != 1024:
        bad.append(f"{video_id}: shape {tuple(value.shape)}")
if bad:
    sys.exit(f"{len(bad)} of {len(video_ids)} videos have no usable features, e.g. {bad[:5]}")
print(f"Features for all {len(video_ids)} videos are in {features}")
PY
