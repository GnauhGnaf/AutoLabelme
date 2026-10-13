"""Annotate every frame of a clip, with one name per entity for the whole clip.

Per frame: coarse (Mask R-CNN) -> select targets -> SAM2 refine -> background gate,
within-frame dedup and polygonize -> tracker. The three models are built once for the
entire clip rather than once per frame, which is what makes a 124-frame run tractable.

Names cannot be known per frame - a track has to be seen MIN_HITS times before it earns a
number - so frames are held in memory as (track id, polygon) and written out at the end
against the final name map. Only polygons are held, never masks: 124 frames of 1024x768
bool masks would be ~1 GB, polygons are ~1 MB.

  gpu env: python video_annotate.py <video> [--stride N] [--out DIR] [--work DIR] [--limit N]

Writes <out>/frames/<frame:06d>.json  labelme 5.10.1, names stable across the clip
       <out>/ov/<frame:06d>.png      overlay, colour keyed by label name
       <out>/tracks.json             name / class / span / gaps / identity decisions
       <out>/preview.mp4             the overlays, playing at the clip's real-time rate
Extracted frame PNGs go to <work>/frames/ - intermediates, the way the still path uses _out/.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coarse_seg  # noqa: E402 - same directory
import entities as E  # noqa: E402
import masks_to_labelme as m2l  # noqa: E402
import sam2_masks  # noqa: E402
import select_targets  # noqa: E402
import track as T  # noqa: E402
import video_frames  # noqa: E402

SCORE_THR = 0.25
FALLBACK_IOU = 0.55
EPS = 1.0


def _write_preview(ov_paths, out_path, fps):
    """Stitch the overlays into an mp4. Unavailable codecs must not lose the JSONs."""
    if not ov_paths:
        return None
    h, w = E.load_bgr(ov_paths[0]).shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
    if not vw.isOpened():
        print(f"  ! no mp4 encoder available, skipping {out_path}")
        return None
    for p in ov_paths:
        vw.write(E.load_bgr(p))
    vw.release()
    return out_path


def run(video, out_dir, work_dir, stride=1, limit=None, model_name="sam2:latest",
        verbose=True):
    """Annotate a clip end to end. Returns the tracks.json dict."""
    info = video_frames.probe(video)
    frames = video_frames.extract(video, os.path.join(work_dir, "frames"), stride)
    if limit:
        frames = frames[:limit]
    if not frames:
        raise SystemExit(f"no frames extracted from {video!r}")
    W, H = info["width"], info["height"]
    print(f"{os.path.basename(video)}: {info['frames']} frames @ {info['fps']:.1f} fps "
          f"{W}x{H} -> {len(frames)} sampled (stride={stride})")

    cmodel, smodel = coarse_seg.get_model(), sam2_masks.get_model(model_name)
    tracker = T.Tracker(W, H)
    records = {}          # frame index -> [(tid, points, class)]

    for k, fr in enumerate(frames):
        path = fr["path"]
        im = Image.open(path).convert("RGB")
        bgr = np.array(im)[:, :, ::-1].copy()      # cv2 needs a contiguous BGR array
        co = coarse_seg.run(path, None, SCORE_THR, model=cmodel, dump=False)
        se = select_targets.run(co, None, None, dump=False, verbose=False)
        sa = sam2_masks.run(se["targets"], se["refs"], path, out_prefix=None,
                            model_name=model_name, fallback_iou=FALLBACK_IOU,
                            model=smodel, arr=np.array(im), dump=False, verbose=False)
        items = m2l.build_items(np.array(sa["masks"], bool), sa["meta"], EPS, verbose=False)
        asg = tracker.step(k, items, bgr)      # k = sample ordinal, not fr["index"]
        records[k] = [(tid, it["points"], it["name"]) for tid, it in asg]
        dropped = len(items) - len({tid for tid, _ in asg})
        if verbose:
            print(f"  [{k + 1}/{len(frames)}] frame {fr['index']:06d}: "
                  f"{len(items)} entities ({dropped} dropped as duplicates)", flush=True)

    labels = tracker.finalize()

    # --- write the frames, now that the names exist ------------------------
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    ov_paths, duplicated = [], []
    for k, fr in enumerate(frames):
        recs = [(tid, pts, nm) for tid, pts, nm in records[k] if tid in labels]
        shapes = E.order_shapes([{"name": nm, "points": pts} for _, pts, nm in recs],
                                [labels[tid] for tid, _, _ in recs])
        names = [s["label"] for s in shapes]
        if len(set(names)) != len(names):
            # One name cannot be two shapes in one frame. Hungarian is one-to-one and
            # merges never join two tracks that co-occur, so this is unreachable - it is
            # here as a tripwire, not as a fallback.
            duplicated.append((fr["index"], names))
        out_json = os.path.join(out_dir, "frames", f"{fr['index']:06d}.json")
        im = Image.open(fr["path"]).convert("RGB")
        E.write_labelme(im, fr["path"], shapes, out_json)
        # ov/ sits beside frames/, not inside it. E.ov_path would nest it under frames/,
        # because it makes ov/ a sibling of the json rather than of the json's directory.
        ov = os.path.join(out_dir, "ov", f"{fr['index']:06d}.png")
        ov_paths.append(E.render_overlay(im, shapes, ov))
    if duplicated:
        raise SystemExit(f"duplicate labels within a frame: {duplicated[:3]}")

    # --- track table + preview --------------------------------------------
    frame_of = [fr["index"] for fr in frames]
    tracks = tracker.summary(frame_of=frame_of)
    result = {
        "video": os.path.basename(video), "source_frames": info["frames"],
        "fps": round(info["fps"], 3), "stride": stride, "sampled_frames": len(frames),
        "model": model_name,
        "params": {"MIN_HITS": T.MIN_HITS, "LOST_MAX": T.LOST_MAX, "REID_SIM": T.REID_SIM,
                   "IOU_GATE": T.IOU_GATE, "DIST_GATE": T.DIST_GATE,
                   "MERGE_SIM": T.MERGE_SIM, "CROSS_SIM": T.CROSS_SIM},
        "tentative_tracks_dropped": len([t for t in tracker.tracks if t.hits < T.MIN_HITS]),
        "tracks": tracks,
        # Tracker events carry a SAMPLE ORDINAL, like everything else inside the tracker;
        # map it here so the file speaks source frames throughout. At stride 1 the two
        # coincide, which is exactly why this went unnoticed until a strided run.
        "events": [{"frame": frame_of[f], "kind": k, "message": m}
                   for f, k, m in tracker.events],
    }
    with open(os.path.join(out_dir, "tracks.json"), "w") as fh:
        json.dump(result, fh, indent=1)

    # At stride N the overlays are N frames apart, so play them back N times slower than
    # the source rate to keep the clip the same wall-clock length.
    _write_preview(ov_paths, os.path.join(out_dir, "preview.mp4"), info["fps"] / max(stride, 1))

    print(f"\n{len(tracks)} entities over {len(frames)} frames "
          f"({result['tentative_tracks_dropped']} tentative tracks dropped)")
    print(f"{'name':10s} {'class':12s} {'first':>6s} {'last':>6s} {'seen':>5s}  gaps")
    for t in tracks:
        print(f"{t['label']:10s} {t['class']:12s} {t['first_frame']:6d} {t['last_frame']:6d} "
              f"{t['n_samples']:5d}  {t['gaps'] if t['gaps'] else '-'}")
    for ev in result["events"]:
        print(f"  frame {ev['frame']:06d} {ev['kind']}: {ev['message']}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth frame (the extracted frames keep their real indices)")
    ap.add_argument("--out", default=None, help="output dir (default output/<video stem>/)")
    ap.add_argument("--work", default=None, help="intermediate dir (default _out/<video stem>/)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N sampled frames")
    ap.add_argument("--model", default="sam2:latest")
    a = ap.parse_args()

    stem = os.path.splitext(os.path.basename(a.video))[0]
    # Mirror run_video.sh: a strided run gets its own directory. Sharing one with the full
    # run would let a 16-frame pass silently replace a 124-frame one.
    sfx = "" if a.stride == 1 else f"-s{a.stride}"
    run(a.video, a.out or os.path.join("output", stem + sfx),
        a.work or os.path.join("_out", stem + sfx), a.stride, a.limit, a.model)


if __name__ == "__main__":
    main()
