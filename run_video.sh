#!/usr/bin/env bash
# Video -> one labelme json per frame, with every entity keeping ONE name for the whole
# clip (person1 is the same person in frame 0 and frame 123, and no entity is ever given
# two names in one frame).
#
#   bash run_video.sh pi-image-scene.mp4        # every frame
#   bash run_video.sh pi-image-scene.mp4 8      # every 8th frame (fast iteration)
#   STRIDE=8 bash run_video.sh a.mp4 b.mp4      # several clips, same stride
#   MODEL=sam2:latest bash run_video.sh clip.mp4
#
# An argument may be a path, a bare filename, or just the stem (looked up in input/).
# With no arguments, every video sitting in the repo root is processed.
#
# Reads  : <video>
# Writes : output/<stem>/frames/<frame>.json   one labelme 5.10.1 file per sampled frame
#          output/<stem>/ov/<frame>.png        overlay, colour keyed by name
#          output/<stem>/tracks.json           entity table + identity decisions
#          output/<stem>/preview.mp4           the overlays, at the clip's real-time rate
#          _out/<stem>/frames/*.png            the extracted frames (intermediates)
#          A stride other than 1 replaces <stem> with <stem>-s<N> throughout, so a strided
#          pass cannot silently truncate a full one.
set -e
cd "$(dirname "$0")"
# Same single conda env as run.sh; see deps/.
PY="${PY:-conda run --no-capture-output -n gpu python}"
MODEL="${MODEL:-sam2:latest}"
STRIDE="${STRIDE:-1}"
P=pipeline; WORK=_out
mkdir -p "$WORK"

# A trailing bare integer is the stride, so `run_video.sh clip.mp4 8` works. Only read
# that way when the first argument really is an existing file, so a clip literally named
# "8" would still be processed as a clip.
if [ "$#" -gt 1 ] && [ -e "$1" ]; then
  last="${!#}"
  if [[ "$last" =~ ^[0-9]+$ ]]; then
    STRIDE="$last"
    set -- "${@:1:$#-1}"
  fi
fi

resolve() {           # argument -> path; a path, <stem>.<ext>, or input/<stem>.<ext>
  if [ -f "$1" ]; then printf '%s' "$1"; return 0; fi
  for e in mp4 mov avi mkv webm m4v; do
    if [ -f "$1.$e" ]; then printf '%s' "$1.$e"; return 0; fi
    if [ -f "input/$1.$e" ]; then printf '%s' "input/$1.$e"; return 0; fi
  done
  return 1
}

shopt -s nullglob
if [ "$#" -eq 0 ]; then
  VIDS=(*.mp4 *.mov *.avi *.mkv *.webm *.m4v)
  [ "${#VIDS[@]}" -gt 0 ] || { echo "no videos given and none in the repo root" >&2; exit 1; }
else
  VIDS=("$@")
fi

ran=0
for a in "${VIDS[@]}"; do
  if ! v="$(resolve "$a")"; then
    echo "skip $a (no such file, and no input/$a.<ext>)" >&2
    continue
  fi
  # Strip directory and extension: pi-image-scene.mp4 -> pi-image-scene
  b="${v##*/}"; stem="${b%.*}"
  # A strided run writes far fewer frames. Sharing a directory with the full run would
  # leave it silently truncated - 124 frame jsons quietly replaced by 16 - so a stride
  # other than 1 gets its own directory and the two can never overwrite each other.
  case "$STRIDE" in 1) SFX="";; *) SFX="-s$STRIDE";; esac
  echo "== $stem$SFX =="
  $PY $P/video_annotate.py "$v" --stride "$STRIDE" --model "$MODEL" \
      --out "output/$stem$SFX" --work "$WORK/$stem$SFX"
  ran=$((ran + 1))
done

# Nothing matched at all is an error; a partial match is not (mirrors run.sh).
[ "$ran" -gt 0 ] || { echo "none of the given videos resolved" >&2; exit 1; }
