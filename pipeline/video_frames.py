"""Deterministic frame extraction for the video path.

cv2.VideoCapture rather than an ffmpeg binary: no dependency the repo has not already
declared, and no PATH assumptions on Windows.

  gpu env: python video_frames.py <video> <out_dir> [stride]

Writes <out_dir>/<frame:06d>.png for every stride-th frame, starting at frame 0.
"""
import os
import sys

import cv2


def extract(video_path, out_dir, stride=1):
    """Decode ``video_path``, writing every stride-th frame as PNG. Returns frame records.

    ``index`` is the ORIGINAL frame number, not the position within the extracted subset -
    so STRIDE=8 still yields 000000, 000008, ... and a strided run stays directly
    comparable with a full one. Frames are grabbed but only decoded when selected, which
    is most of the saving on a strided pass.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames, idx = [], 0
    while cap.grab():
        if idx % stride == 0:
            ok, im = cap.retrieve()
            if not ok:
                break
            path = os.path.join(out_dir, f"{idx:06d}.png")
            # imencode + write bytes instead of cv2.imwrite: imwrite goes through the ANSI
            # code page on Windows and silently fails on a non-ASCII output path.
            ok, buf = cv2.imencode(".png", im)
            if not ok:
                raise RuntimeError(f"cannot encode frame {idx}")
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            frames.append({"index": idx, "path": path,
                           "time": round(idx / fps, 3) if fps else None})
        idx += 1
    cap.release()

    print(f"{len(frames)} frames extracted from {total} (stride={stride}) -> {out_dir}")
    return frames


def probe(video_path):
    """{"fps", "frames", "width", "height"} for a clip, without decoding it."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path!r}")
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def main():
    video_path, out_dir = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    extract(video_path, out_dir, stride)


if __name__ == "__main__":
    main()
