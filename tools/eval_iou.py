"""Best-match polygon IoU between a predicted labelme json and a GT labelme json.
Matching is by highest overlap (greedy), independent of the numbering order.

Usage: eval_iou.py <pred.json> <gt.json>
"""
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "pipeline"))
from entities import rasterize  # noqa: E402 - needs the pipeline dir on sys.path first


def main():
    pred = json.load(open(sys.argv[1]))
    gt = json.load(open(sys.argv[2]))
    W, H = gt["imageWidth"], gt["imageHeight"]
    pm = {s["label"]: rasterize(s["points"], W, H) for s in pred["shapes"]}
    gm = {s["label"]: rasterize(s["points"], W, H) for s in gt["shapes"]}
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
