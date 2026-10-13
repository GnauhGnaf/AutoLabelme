#!/usr/bin/env bash
# Self-contained input/ -> output/ runner using only the code in pipeline/.
#   bash run.sh                    # every image in input/
#   bash run.sh 0486 0487          # only these ids
#   bash run.sh 0486.jpg           # a filename works too
#   MODEL=sam2:latest bash run.sh
# Reads  : input/<id>.<ext>, ext in IMAGE_EXTS below (case-insensitive)
# Writes : output/<id>.json + preview output/ov/<id>.png  (intermediates in _out/)
set -e
cd "$(dirname "$0")"
# All four stages run in the single conda env "gpu" (see deps/), which carries
# the SAM2.1 / osam stack as well as torch. Override if installed differently:
#   PY=/path/to/python bash run.sh
PY="${PY:-conda run --no-capture-output -n gpu python}"
P=pipeline; IN=input; OUT=output; WORK=_out
MODEL="${MODEL:-sam2:latest}"
# Keep in sync with IMAGE_EXTS in pipeline/entities.py
IMAGE_EXTS=(png jpg jpeg bmp webp tif tiff)
mkdir -p "$OUT" "$WORK"

shopt -s nullglob

is_image() {          # $1 = an extension
  local e="${1,,}" x
  for x in "${IMAGE_EXTS[@]}"; do
    [ "$e" = "$x" ] && return 0
  done
  return 1
}

[ -d "$IN" ] || { echo "no $IN/ directory" >&2; exit 1; }

# Index input/ once: id -> path. The glob is locale-sorted, so IDS keeps a stable order.
declare -A SRC=() AMBIG=()
IDS=()
for f in "$IN"/*; do
  [ -f "$f" ] || continue
  b="${f##*/}"
  is_image "${b##*.}" || continue
  id="${b%.*}"
  [ -n "$id" ] || continue                  # a bare '.png' has no stem
  if [ -n "${SRC[$id]+set}" ]; then
    AMBIG[$id]="${AMBIG[$id]:-${SRC[$id]}} $f"
  else
    IDS+=("$id")
  fi
  SRC[$id]="$f"
done

# Selection: no args = everything; args are ids or filenames.
if [ "$#" -gt 0 ]; then
  SEL=()
  for a in "$@"; do
    a="${a##*/}"                            # tolerate being handed a path
    if [ -n "${SRC[$a]+set}" ]; then id="$a"      # an exact id wins...
    elif is_image "${a##*.}"; then id="${a%.*}"   # ...else read it as a filename
    else id="$a"; fi
    if [ -z "${SRC[$id]+set}" ]; then
      echo "skip $a (no $IN/$id.<ext>)"
      continue
    fi
    SEL+=("$id")
  done
else
  SEL=("${IDS[@]}")
fi

if [ "${#SEL[@]}" -eq 0 ]; then
  if [ "${#IDS[@]}" -eq 0 ]; then
    echo "no images in $IN/ (looked for: ${IMAGE_EXTS[*]})" >&2
  else
    echo "none of the given ids matched an image in $IN/" >&2
  fi
  exit 1
fi

for id in "${SEL[@]}"; do
  # Two inputs sharing a stem would both write output/<id>.json, so refuse rather than
  # let the second silently overwrite the first.
  if [ -n "${AMBIG[$id]+set}" ]; then
    echo "error: id '$id' is ambiguous, several inputs share that stem:" >&2
    echo "  ${AMBIG[$id]}" >&2
    exit 1
  fi
  IMG="${SRC[$id]}"
  echo "== $id =="
  $PY $P/coarse_seg.py "$IMG" "$WORK/$id" 0.25
  $PY $P/select_targets.py "$WORK/$id" "$WORK/$id"
  $PY $P/sam2_masks.py "$WORK/${id}_targets.json" "$WORK/${id}_ref.npz" "$IMG" "$WORK/$id" "$MODEL" 0.55 2>/dev/null
  $PY $P/masks_to_labelme.py "$WORK/${id}_masks.npz" "$WORK/${id}_masks.json" "$IMG" "$OUT/$id.json" 1.0
done
