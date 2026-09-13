"""Step C (gpu env, cv2): polygonize refined masks -> labelme json + overlay.

- Drops background targets whose refined mask still overlaps people a lot (leaky masks).
- Keeps at most 2 background entities (largest by mask area).
- Naming: '<class><n>', n assigned left-to-right by centroid x, per class.

  gpu env: python masks_to_labelme.py <prefix>_masks.npz <prefix>_masks.json <image> <out.json> [eps]
"""
import sys, json, base64
import numpy as np
import cv2
import colorsys
from PIL import Image, ImageDraw, ImageFont

BG_MAX = 2
BG_PERSON_OVERLAP = 0.30


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


def main():
    npz, metaj, img_path, out_path = sys.argv[1:5]
    eps = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

    masks = np.load(npz)["masks"]
    meta = json.load(open(metaj))
    im = Image.open(img_path).convert("RGB")
    W, H = im.size

    person = np.any([m for m, md in zip(masks, meta) if md["name"] == "person"], axis=0) \
        if any(md["name"] == "person" for md in meta) else np.zeros((H, W), bool)

    # background gating: drop leaky masks, keep the BG_MAX largest survivors
    bg_idx = [k for k, md in enumerate(meta) if md["src"] == "bg"]
    good_bg = []
    for k in bg_idx:
        m = masks[k]
        ov = float(np.logical_and(m, person).sum()) / max(1, int(m.sum()))
        if ov > BG_PERSON_OVERLAP:
            print(f'  x bg {meta[k]["name"]}: dropped (person overlap {ov:.2f})')
            continue
        good_bg.append((int(m.sum()), k))
    good_bg.sort(reverse=True)
    keep_bg = {k for _, k in good_bg[:BG_MAX]}
    for area, k in good_bg[BG_MAX:]:
        print(f'  x bg {meta[k]["name"]}: dropped (not among top {BG_MAX} by area={area})')

    items = []
    for k, (m, md) in enumerate(zip(masks, meta)):
        if md["src"] == "bg" and k not in keep_bg:
            continue
        poly = polygonize(m, eps, bridge=(md["name"] == "person"))
        if poly:
            items.append({"name": md["name"], "points": poly})

    shapes = []
    for cls in sorted({it["name"] for it in items}, key=lambda c: (c != "person", c)):
        group = sorted([it for it in items if it["name"] == cls],
                       key=lambda it: np.mean([p[0] for p in it["points"]]))
        for n, it in enumerate(group, 1):
            shapes.append({"label": f"{cls}{n}", "points": it["points"]})

    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    json.dump({
        "version": "5.10.1", "flags": {},
        "shapes": [{"label": s["label"], "points": [[round(x, 1), round(y, 1)] for x, y in s["points"]],
                    "group_id": None, "description": "", "shape_type": "polygon", "flags": {}, "mask": None}
                   for s in shapes],
        "imagePath": img_path.replace("\\", "/").split("/")[-1],
        "imageData": b64, "imageHeight": H, "imageWidth": W,
    }, open(out_path, "w"), indent=2)

    ov = im.convert("RGBA")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    cols = {}
    for i, s in enumerate(shapes):
        cols[s["label"]] = tuple(int(v * 255) for v in colorsys.hsv_to_rgb(i / max(1, len(shapes)), 0.78, 0.95))
        dr.polygon([tuple(p) for p in s["points"]], fill=cols[s["label"]] + (105,), outline=cols[s["label"]] + (255,))
    ov = Image.alpha_composite(ov, lay)
    dr = ImageDraw.Draw(ov)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    for s in shapes:
        p = np.array(s["points"], float)
        cx, cy = p[:, 0].mean(), p[:, 1].min()
        dr.rectangle([cx - 2, cy - 30, cx + 8 + 13 * len(s["label"]), cy - 2], fill=cols[s["label"]] + (235,))
        dr.text((cx + 2, cy - 29), s["label"], fill=(0, 0, 0, 255), font=font)
    ov.convert("RGB").save(out_path.replace(".json", "_ov.png"))
    print(f"wrote {out_path}: {[s['label'] for s in shapes]}")


if __name__ == "__main__":
    main()
