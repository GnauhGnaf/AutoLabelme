"""Coarse pass: COCO Mask R-CNN -> instance masks (kept for downstream selection).
Low score threshold on purpose: low-confidence objects are decided later by visual review.

  gpu env: python coarse_seg.py <image> <out_prefix> [score_thr]
Outputs <prefix>_masks.npz (bool N,H,W), <prefix>_inst.json, <prefix>_boxes.png

The video driver imports run() instead of shelling out, which is why the network lives
behind get_model(): building it once per process rather than once per frame is the
difference between a ~30 minute and a several hour 124-frame run.
"""
import sys, os, json
import numpy as np
import cv2
import torch
import torchvision
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image, ImageDraw, ImageFont

WEIGHTS = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
_MODEL = None


def get_model():
    """The COCO Mask R-CNN, built at most once per process. Weights are already cached."""
    global _MODEL
    if _MODEL is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=WEIGHTS).eval().to(dev)
    return _MODEL


def polygonize(mask, eps=2.0):
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None
    return cv2.approxPolyDP(c, eps, True).reshape(-1, 2).tolist()


def _dump_boxes(im, inst, out_path):
    canvas = im.copy()
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    rng = np.random.default_rng(0)
    for it in inst:
        col = tuple(int(v) for v in rng.integers(60, 255, 3))
        x0, y0, x1, y1 = it["box"]
        d.rectangle([x0, y0, x1, y1], outline=col, width=2)
        d.text((x0 + 2, y0 + 1), f'{it["label"]} {it["score"]:.2f}', fill=col, font=font)
    canvas.save(out_path)


def run(img_path, out_prefix=None, score_thr=0.25, model=None, dump=True):
    """Coarse-detect one image and return what the next stage needs.

    Returns {"image", "width", "height", "instances", "masks"}: the instance list and the
    bool (N, H, W) mask stack. ``dump`` additionally writes <out_prefix>_masks.npz /
    _inst.json / _boxes.png, i.e. the CLI contract. The video driver passes dump=False -
    it already holds these arrays, and 124 frames of intermediates are pure waste.
    """
    model = get_model() if model is None else model
    dev = next(model.parameters()).device

    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    x = pil_to_tensor(im).float().div(255).to(dev)
    with torch.no_grad():
        out = model([x])[0]

    masks = out["masks"].cpu().numpy()[:, 0]
    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    boxes = out["boxes"].cpu().numpy()
    cats = WEIGHTS.meta["categories"]

    keep = scores >= score_thr
    masks, labels, scores, boxes = masks[keep], labels[keep], scores[keep], boxes[keep]
    inst = [{"label": cats[int(labels[i])], "score": round(float(scores[i]), 3),
             "box": [round(float(v), 1) for v in boxes[i]],
             "points": polygonize(masks[i])} for i in range(len(scores))]

    res = {"image": os.path.basename(img_path), "width": W, "height": H,
           "instances": inst, "masks": (masks > 0.5)}

    if dump and out_prefix:
        np.savez_compressed(out_prefix + "_masks.npz", masks=res["masks"])
        json.dump({k: v for k, v in res.items() if k != "masks"},
                  open(out_prefix + "_inst.json", "w"), indent=1)
        _dump_boxes(im, inst, out_prefix + "_boxes.png")
    print(f"{len(inst)} instances (score>={score_thr})")
    return res


def main():
    img_path, out_prefix = sys.argv[1], sys.argv[2]
    score_thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    run(img_path, out_prefix, score_thr)


if __name__ == "__main__":
    main()
