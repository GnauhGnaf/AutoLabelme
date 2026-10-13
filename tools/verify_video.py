"""Pure-code verification of a video annotation run. No model, no agent, no eyeballing.

  gpu env: python tools/verify_video.py <out_dir> [--compare <other_out_dir>]

Checks:
  1 structure    every frame json is labelme 5.10.1, polygon shapes, >=3 points, and its
                 embedded image matches imageWidth/imageHeight
  2 duplicates   no two shapes in one frame share a label  (the core requirement)
  3 continuity   an identity swap moves TWO entities at the same step, so that - not a
                 single fast-moving object - is what this looks for
  4 cross-name   two same-class entities under different names never overlap heavily; if
                 they did, the within-frame dedup missed a duplicate
  5 table        name / class / span / samples / gaps
  6 determinism  with --compare: two runs of the same stride produce identical output

Exit status is 1 if any check fails, so this can gate a run.
"""
import argparse
import base64
import collections
import io
import json
import os
import sys

import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "pipeline"))
import entities as E  # noqa: E402 - needs the pipeline dir on sys.path first

SWAP_DIST = 0.25       # fraction of the diagonal that counts as a jump
SWAP_MIN = 2           # entities that must jump together for it to be a swap
DUP_IOU = E.DEDUP_IOU_SAME
FAIL = []


def fail(msg):
    FAIL.append(msg)
    print(f"  FAIL {msg}")


def load_run(out_dir):
    """-> (frames, shapes_by_frame, meta) with masks already rasterized."""
    fdir = os.path.join(out_dir, "frames")
    if not os.path.isdir(fdir):
        raise SystemExit(f"no frames/ in {out_dir!r}")
    files = sorted(f for f in os.listdir(fdir) if f.endswith(".json"))
    frames, shapes = [], {}
    for name in files:
        d = json.load(open(os.path.join(fdir, name), encoding="utf-8"))
        idx = int(name[:-5])
        frames.append((idx, d))
        shapes[idx] = d["shapes"]
    return frames, shapes


def check_structure(frames):
    print(f"\n[1] structure ({len(frames)} frames)")
    for idx, d in frames:
        tag = f"{idx:06d}"
        if d.get("version") != "5.10.1":
            fail(f"{tag}: version {d.get('version')!r}")
        W, H = d["imageWidth"], d["imageHeight"]
        for s in d["shapes"]:
            if s["shape_type"] != "polygon":
                fail(f"{tag} {s['label']}: shape_type {s['shape_type']!r}")
            if len(s["points"]) < 3:
                fail(f"{tag} {s['label']}: {len(s['points'])} points")
        try:
            im = Image.open(io.BytesIO(base64.b64decode(d["imageData"])))
        except Exception as e:                       # noqa: BLE001 - any malformed blob
            fail(f"{tag}: imageData does not decode ({e})")
            continue
        if im.size != (W, H):
            fail(f"{tag}: imageData {im.size} != {W}x{H}")
    print("  ok" if not FAIL else "")


def check_duplicates(frames):
    print("\n[2] no duplicate labels within a frame")
    bad = 0
    for idx, d in frames:
        names = [s["label"] for s in d["shapes"]]
        dups = [n for n, c in collections.Counter(names).items() if c > 1]
        if dups:
            bad += 1
            fail(f"{idx:06d}: repeated {dups}")
    print(f"  ok - {len(frames)} frames, {bad} with duplicates")


def check_cross_name(frames, shapes):
    """Same class, different names, overlapping masks - a dedup miss.

    Cross-class overlaps are reported but NOT failed: a legitimate nesting looks exactly
    like a duplicate (a phone inside a bag, a book inside a person's silhouette), and the
    two cannot be told apart from geometry alone. Only same-class overlap is unambiguous
    enough to fail on.
    """
    print("\n[4] same-class entities under different names stay apart")
    worst = collections.defaultdict(float)
    cross = collections.defaultdict(float)
    W = frames[0][1]["imageWidth"]
    H = frames[0][1]["imageHeight"]
    for idx, _ in frames:
        ss = shapes[idx]
        ras = [(s["label"], E.rasterize(s["points"], W, H)) for s in ss]
        for i in range(len(ras)):
            for j in range(i + 1, len(ras)):
                (la, ma), (lb, mb) = ras[i], ras[j]
                ca, cb = E.split_label(la)[0], E.split_label(lb)[0]
                v = E.iou(ma, mb)
                if ca == cb:
                    worst[ca] = max(worst[ca], v)
                    if v > DUP_IOU:
                        fail(f"{idx:06d}: {la} and {lb} overlap IoU={v:.2f}")
                elif "person" not in (ca, cb):
                    cross[f"{ca}/{cb}"] = max(cross[f"{ca}/{cb}"], v)
    for cls in sorted(worst):
        print(f"  max IoU within class {cls:10s} = {worst[cls]:.3f}  (flag above {DUP_IOU})")
    if not worst:
        print("  ok - no same-class overlap at all")
    if cross:
        top = sorted(cross.items(), key=lambda kv: -kv[1])[:5]
        print(f"  non-person cross-class overlaps (reported, not failed): "
              f"{[(k, round(v, 2)) for k, v in top]}")


def check_continuity(frames, shapes):
    """An identity swap makes two entities jump at the SAME step. Look for that."""
    print("\n[3] identity continuity")
    W = frames[0][1]["imageWidth"]
    H = frames[0][1]["imageHeight"]
    diag = float(np.hypot(W, H))
    cents = {}
    for idx, _ in frames:
        cents[idx] = {s["label"]: E.centroid(s["points"]) for s in shapes[idx]}
    idxs = [i for i, _ in frames]

    worst = collections.defaultdict(float)
    swaps = []
    for a, b in zip(idxs, idxs[1:]):
        jumps = []
        for lab in set(cents[a]) & set(cents[b]):
            (x0, y0), (x1, y1) = cents[a][lab], cents[b][lab]
            d = float(np.hypot(x1 - x0, y1 - y0)) / diag
            worst[lab] = max(worst[lab], d)
            if d > SWAP_DIST:
                jumps.append((lab, d))
        if len(jumps) >= SWAP_MIN:
            swaps.append((a, b, jumps))
    for a, b, jumps in swaps:
        fail(f"{a:06d}->{b:06d}: {len(jumps)} entities jumped at once "
             f"{[(l, round(d, 2)) for l, d in jumps]} - looks like a swap")
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:4]
    print(f"  largest single-step moves (fraction of diagonal): "
          f"{[(l, round(d, 2)) for l, d in top]}")
    print(f"  swap-flagged transitions: {len(swaps)}")
    print("  (a swap needs >= %d entities to jump > %.2f together)" % (SWAP_MIN, SWAP_DIST))


def check_table(out_dir):
    print("\n[5] entity table")
    p = os.path.join(out_dir, "tracks.json")
    if not os.path.exists(p):
        fail("no tracks.json")
        return
    d = json.load(open(p))
    print(f"  {d['video']}: {d['source_frames']} frames @ {d['fps']}fps, "
          f"stride {d['stride']} -> {d['sampled_frames']} sampled")
    print(f"  {len(d['tracks'])} entities, {d['tentative_tracks_dropped']} tentative dropped")
    print(f"  {'name':10s} {'class':12s} {'first':>6s} {'last':>6s} {'seen':>5s}  gaps")
    for t in d["tracks"]:
        print(f"  {t['label']:10s} {t['class']:12s} {t['first_frame']:6d} "
              f"{t['last_frame']:6d} {t['n_samples']:5d}  {t['gaps'] if t['gaps'] else '-'}")
    for ev in d["events"]:
        print(f"  frame {ev['frame']:06d} {ev['kind']}: {ev['message']}")
    labels = [t["label"] for t in d["tracks"]]
    if len(set(labels)) != len(labels):
        fail("tracks.json has a repeated label")


def check_determinism(out_dir, other):
    print(f"\n[6] determinism: {out_dir} vs {other}")
    a, b = load_run(out_dir), load_run(other)
    if [i for i, _ in a[0]] != [i for i, _ in b[0]]:
        fail("different frame sets")
        return
    diff = 0
    for (ia, da), (ib, db) in zip(a[0], b[0]):
        if da["shapes"] != db["shapes"]:
            diff += 1
            fail(f"{ia:06d}: shapes differ")
    ta = json.load(open(os.path.join(out_dir, "tracks.json")))
    tb = json.load(open(os.path.join(other, "tracks.json")))
    if ta["tracks"] != tb["tracks"]:
        fail("tracks.json entity table differs")
    print(f"  {len(a[0])} frame jsons compared, {diff} differ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--compare", default=None, help="a second run, for the determinism check")
    a = ap.parse_args()

    frames, shapes = load_run(a.out_dir)
    check_structure(frames)
    check_duplicates(frames)
    check_continuity(frames, shapes)
    check_cross_name(frames, shapes)
    check_table(a.out_dir)
    if a.compare:
        check_determinism(a.out_dir, a.compare)

    print("\n" + ("VERIFY FAIL:\n  " + "\n  ".join(FAIL) if FAIL else "VERIFY PASS"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
