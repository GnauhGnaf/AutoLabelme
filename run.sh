#!/usr/bin/env bash
# Self-contained input/ -> output/ runner using only the code in pipeline/.
#   bash run.sh              # run every image in input/
#   bash run.sh 0486 0487    # run only these ids
#   MODEL=sam2:latest bash run.sh
# Reads  : input/<id>.png
# Writes : output/<id>.json   (intermediates in _out/)
set -e
cd "$(dirname "$0")"
# All four stages run in the single conda env "gpu" (see deps/), which carries
# the SAM2.1 / osam stack as well as torch. Override if installed differently:
#   PY=/path/to/python bash run.sh
PY="${PY:-conda run --no-capture-output -n gpu python}"
P=pipeline; IN=input; OUT=output; WORK=_out
MODEL="${MODEL:-sam2:latest}"
mkdir -p "$OUT" "$WORK"

if [ "$#" -gt 0 ]; then IDS="$*"; else IDS=$(ls "$IN" | sed -n 's/\.png$//p'); fi

for ID in $IDS; do
  IMG="$IN/$ID.png"
  if [ ! -f "$IMG" ]; then echo "skip $ID (no image in $IN/)"; continue; fi
  echo "== $ID =="
  $PY $P/coarse_seg.py "$IMG" "$WORK/$ID" 0.25
  $PY $P/select_targets.py "$WORK/$ID" "$WORK/$ID"
  $PY $P/sam2_masks.py "$WORK/${ID}_targets.json" "$WORK/${ID}_ref.npz" "$IMG" "$WORK/$ID" "$MODEL" 0.55 2>/dev/null
  $PY $P/masks_to_labelme.py "$WORK/${ID}_masks.npz" "$WORK/${ID}_masks.json" "$IMG" "$OUT/$ID.json" 1.0
done
