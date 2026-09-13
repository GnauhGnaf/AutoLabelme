#!/usr/bin/env bash
# Self-contained input/ -> output/ runner using only the code in label/.
#   bash label/run.sh              # run every image in label/input/
#   bash label/run.sh 0486 0487    # run only these ids
#   MODEL=sam2:latest bash label/run.sh
# Reads  : label/input/<id>.png
# Writes : label/output/<id>.json   (intermediates in label/_out/)
set -e
cd "$(dirname "$0")"
# Interpreters for the two envs (see deps/). Defaults use conda env names
# "gpu" and "labelme"; override if you install them differently:
#   GPU_PY=/path/to/python LME_PY=/path/to/python bash run.sh
GPU_PY="${GPU_PY:-conda run --no-capture-output -n gpu python}"
LME_PY="${LME_PY:-conda run --no-capture-output -n labelme python}"
P=pipeline; IN=input; OUT=output; WORK=_out
MODEL="${MODEL:-sam2:latest}"
mkdir -p "$OUT" "$WORK"

if [ "$#" -gt 0 ]; then IDS="$*"; else IDS=$(ls "$IN" | sed -n 's/\.png$//p'); fi

for ID in $IDS; do
  IMG="$IN/$ID.png"
  if [ ! -f "$IMG" ]; then echo "skip $ID (no image in $IN/)"; continue; fi
  echo "== $ID =="
  $GPU_PY $P/coarse_seg.py "$IMG" "$WORK/$ID" 0.25
  $GPU_PY $P/select_targets.py "$WORK/$ID" "$WORK/$ID"
  $LME_PY $P/sam2_masks.py "$WORK/${ID}_targets.json" "$WORK/${ID}_ref.npz" "$IMG" "$WORK/$ID" "$MODEL" 0.55 2>/dev/null
  $GPU_PY $P/masks_to_labelme.py "$WORK/${ID}_masks.npz" "$WORK/${ID}_masks.json" "$IMG" "$OUT/$ID.json" 1.0
done
