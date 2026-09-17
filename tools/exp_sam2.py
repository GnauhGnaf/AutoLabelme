"""Experiment: compare SAM2 prompt configs for refining coarse instances, scored by IoU vs GT.
Runs entirely in the gpu env (PIL rasterization, no cv2).

  python exp_sam2.py <coarse.json> <image> <gt_labelme.json> [model]
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw

from osam import apis, types

BAG = {"handbag", "backpack", "suitcase", "bag", "tote"}
KEEP = {"person": "person", **{k: "bag" for k in BAG}}


def raster(points, W, H):
    m = Image.new("1", (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in points], fill=1)
    return np.array(m, bool)


def erode(m, it=1):
    for _ in range(it):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m


def dilate(m, it=1):
    for _ in range(it):
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return m


def sample(mask, k, rng):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False)
    return [[float(xs[i]), float(ys[i])] for i in idx]


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    coarse = json.load(open(sys.argv[1]))
    img_path = sys.argv[2]
    gt = json.load(open(sys.argv[3]))
    model_name = sys.argv[4] if len(sys.argv) > 4 else "sam2:latest"

    arr = np.array(Image.open(img_path).convert("RGB"))
    H, W = arr.shape[:2]
    gt_masks = {s["label"]: raster(s["points"], W, H) for s in gt["shapes"]}
    inst = [i for i in coarse["instances"] if i["label"] in KEEP]

    model = apis.get_model_type_by_name(model_name)()
    emb = model.encode_image(arr)
    rng = np.random.default_rng(0)

    def run(pts, labels):
        pr = types.Prompt(points=np.array(pts, float), point_labels=np.array(labels, np.int32))
        return apis.generate(types.GenerateRequest(model=model_name, image_embedding=emb, prompt=pr)).annotations[0].mask

    configs = {}
    for i in inst:
        ref = raster(i["points"], W, H)
        db = [int(round(v)) for v in i["box"]]
        p = np.array(i["points"], float)
        tb = [int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())]

        inner = erode(ref, 2) if erode(ref, 2).sum() > 20 else ref
        pos = sample(inner, 4, rng)
        ring = np.zeros((H, W), bool)
        ring[max(0, db[1]):db[3], max(0, db[0]):db[2]] = True
        neg = sample(ring & ~dilate(ref, 3), 8, rng)

        def add(name, box, extra_pos, extra_neg):
            pts = [box[0:2], box[2:4]] + extra_pos + extra_neg
            lab = [2, 3] + [1] * len(extra_pos) + [0] * len(extra_neg)
            configs.setdefault(name, []).append(run(pts, lab))

        add("det_box", db, [], [])
        add("tight_box", tb, [], [])
        add("tight_box+pos", tb, pos, [])
        add("det_box+pos", db, pos, [])
        add("det_box+pos+neg", db, pos, neg)
        configs.setdefault("coarse", []).append(ref)

    print(f"{'config':18s} " + " ".join(f"{g:>7s}" for g in gt_masks) + f" {'mean':>7s}")
    for name, masks in configs.items():
        per = [max(iou(gm, m) for m in masks) for gm in gt_masks.values()]
        print(f"{name:18s} " + " ".join(f"{v:7.3f}" for v in per) + f" {np.mean(per):7.3f}")


if __name__ == "__main__":
    main()
