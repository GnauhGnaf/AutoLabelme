"""Best-match polygon IoU between a predicted labelme json and a GT labelme json.
Matching is by highest overlap (greedy), independent of the numbering order.

Usage: eval_iou.py <pred.json> <gt.json>
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw


def raster(shape, W, H):
    m = Image.new("1", (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in shape["points"]], fill=1)
    return np.array(m, dtype=bool)


def main():
    pred = json.load(open(sys.argv[1]))
    gt = json.load(open(sys.argv[2]))
    W, H = gt["imageWidth"], gt["imageHeight"]
    pm = {s["label"]: raster(s, W, H) for s in pred["shapes"]}
    gm = {s["label"]: raster(s, W, H) for s in gt["shapes"]}
    print(f"{'GT':8s} {'best pred':10s} {'IoU':>6s}")
    ious = []
    for gl, g in gm.items():
        best, bl = 0.0, None
        for pl, p in pm.items():
            inter = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            iou = inter / union if union else 0.0
            if iou > best:
                best, bl = iou, pl
        ious.append(best)
        print(f"{gl:8s} {str(bl):10s} {best:6.3f}")
    print(f"mean IoU = {np.mean(ious):.3f}   n={len(ious)}")


if __name__ == "__main__":
    main()
