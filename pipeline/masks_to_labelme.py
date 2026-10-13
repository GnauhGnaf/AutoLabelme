"""Step C (gpu env, cv2): polygonize refined masks -> labelme json + overlay.

- Drops background targets whose refined mask still overlaps people a lot (leaky masks).
- Keeps at most 2 background entities (largest by mask area).
- Collapses duplicate masks for the same physical object (entities.dedup).
- Naming: '<class><n>', n assigned left-to-right by centroid x, per class.
- Overlay goes to the sibling ov/ directory: output/0486.json -> output/ov/0486.png

  gpu env: python masks_to_labelme.py <prefix>_masks.npz <prefix>_masks.json <image> <out.json> [eps]
"""
import json
import sys

import cv2
import numpy as np
from PIL import Image

import entities as E

BG_MAX = 2
BG_PERSON_OVERLAP = 0.30
# How much the coarse/'bg' provenance matters when picking a survivor among duplicates.
SRC_PRIORITY = {"manual": 3, "det": 2, "bg": 1}


def polygonize(mask, eps, bridge=False):
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    if bridge and len(cnts) > 1:
        # a person occluded across the body splits into stacked components; fill the
        # vertical gap between them (limited to their x-overlap) so the person is one polygon
        areas = [cv2.contourArea(c) for c in cnts]
        boxes = [cv2.boundingRect(c) for c, a in zip(cnts, areas) if a >= 0.15 * max(areas)]
        boxes.sort(key=lambda b: b[1])
        for (ax, ay, aw, ah), (bx, by, bw, bh) in zip(boxes, boxes[1:]):
            gx0, gx1 = max(ax, bx), min(ax + aw, bx + bw)
            if gx1 > gx0 and by > ay + ah:
                m[ay + ah:by + 1, gx0:gx1 + 1] = 1
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None
    return cv2.approxPolyDP(c, eps, True).reshape(-1, 2).tolist()


def trust(md):
    """How much we trust a refined mask, used to choose the survivor among duplicates.

    A SAM2-refined 'det' mask beats a mask that fell back to the coarse one, which beats
    a background entity; ties break on the SAM2/coarse agreement, then on area via
    entities._rank.
    """
    refined = 0.0 if md.get("used") == "coarse" else 1.0
    return SRC_PRIORITY.get(md.get("src"), 0) * 10 + refined + (md.get("sam_iou_vs_ref") or 0.0)


def build_items(masks, meta, eps=1.0, verbose=True):
    """Background gating -> polygonize -> within-frame dedup.

    Returns [{"name", "mask", "points"}] for one frame. Naming is deliberately NOT done
    here: the still-image path names by centroid (``order_shapes`` with labels=None), while
    the video path names from the tracker so an identity outlives the frame. Both consume
    this list, which is why the gate and the dedup live in one place.

    Polygonize runs BEFORE dedup, and the emitted ``mask`` is the polygon rasterized back.
    That ordering is the fix for a real miss: polygonize keeps only the largest contour, so
    it changes how much two entities overlap. Deduping raw masks first let a single bag out
    twice - the detector's bag+strap mask (a coarse fallback) and the SAM2 bag-body mask
    were 0.79 contained as masks but 0.92 as polygons, straddling the 0.85 threshold. The
    polygon is what gets named and delivered, so the polygon is what dedup must judge; the
    tracker then matches on exactly the geometry the JSON contains.
    """
    say = print if verbose else (lambda *a, **k: None)
    H, W = masks.shape[1:]

    person = np.any([m for m, md in zip(masks, meta) if md["name"] == "person"], axis=0) \
        if any(md["name"] == "person" for md in meta) else np.zeros((H, W), bool)

    # background gating: drop leaky masks, keep the BG_MAX largest survivors
    bg_idx = [k for k, md in enumerate(meta) if md["src"] == "bg"]
    good_bg = []
    for k in bg_idx:
        m = masks[k]
        ov = float(np.logical_and(m, person).sum()) / max(1, int(m.sum()))
        if ov > BG_PERSON_OVERLAP:
            say(f'  x bg {meta[k]["name"]}: dropped (person overlap {ov:.2f})')
            continue
        good_bg.append((int(m.sum()), k))
    good_bg.sort(reverse=True)
    keep_bg = {k for _, k in good_bg[:BG_MAX]}
    for area, k in good_bg[BG_MAX:]:
        say(f'  x bg {meta[k]["name"]}: dropped (not among top {BG_MAX} by area={area})')

    cand = []
    for k, md in enumerate(meta):
        if md["src"] == "bg" and k not in keep_bg:
            continue
        poly = polygonize(masks[k], eps, bridge=(md["name"] == "person"))
        if not poly:
            continue          # too small to be a polygon, so it can never be a shape
        cand.append({"name": md["name"], "mask": E.rasterize(poly, W, H),
                     "score": trust(md), "points": poly})

    # collapse candidates that describe the same physical object
    keep, dropped = E.dedup(cand)
    for i, best, why in dropped:
        say(f'  x dup {cand[i]["name"]}: dropped ({why})')
    return [cand[i] for i in keep]


def run(masks, meta, img_path, out_path, eps=1.0, verbose=True):
    """Still-image path: build_items -> centroid naming -> labelme json + ov/ preview."""
    im = Image.open(img_path).convert("RGB")
    items = build_items(masks, meta, eps, verbose)
    shapes = E.order_shapes(items)
    E.write_labelme(im, img_path, shapes, out_path)
    E.render_overlay(im, shapes, E.ov_path(out_path))
    print(f"wrote {out_path}: {[s['label'] for s in shapes]}")
    return shapes


def main():
    npz, metaj, img_path, out_path = sys.argv[1:5]
    eps = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    run(np.load(npz)["masks"], json.load(open(metaj)), img_path, out_path, eps)


if __name__ == "__main__":
    main()
