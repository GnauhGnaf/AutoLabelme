"""Choose annotation targets: every person + every object HELD/carried in hand.

Held test: a portable object is kept when its coarse mask overlaps the (dilated) union of
person masks enough -- i.e. it touches a person's silhouette/hand region.
Manual extras (objects the detector missed) come from an optional extras JSON:
  [{"label": "phone", "box": [x0, y0, x1, y1]}, ...]

  gpu env: python select_targets.py <coarse_prefix> <out_prefix> [extras.json]

Outputs <out_prefix>_targets.json and <out_prefix>_ref.npz (ref masks for the SAM2 guard).
"""
import sys, os, json
import numpy as np
import cv2

# COCO class -> output name. Bags keep the GT convention 'bag'.
PORTABLE = {
    "handbag": "bag", "backpack": "bag", "suitcase": "bag",
    "cell phone": "phone", "cup": "cup", "wine glass": "glass", "bottle": "bottle",
    "book": "book", "scissors": "scissors", "remote": "remote", "laptop": "laptop",
    "keyboard": "keyboard", "mouse": "mouse", "umbrella": "umbrella",
    "toothbrush": "toothbrush", "sports ball": "ball", "tennis racket": "racket",
    "baseball bat": "bat", "bowl": "bowl", "fork": "fork", "knife": "knife", "spoon": "spoon",
    "banana": "banana", "apple": "apple", "sandwich": "sandwich", "donut": "donut", "cake": "cake",
}
OVERLAP_THR = 0.30
DILATE_PX = 25

# Background ("irrelevant") scene entities: prominent furniture/large objects, named by class.
BG_NAME = {
    "potted plant": "plant", "chair": "chair", "bench": "bench", "couch": "sofa",
    "dining table": "table", "bed": "bed", "tv": "tv", "refrigerator": "fridge",
    "microwave": "microwave", "oven": "oven", "toaster": "toaster", "sink": "sink",
    "toilet": "toilet", "vase": "vase", "clock": "clock", "bicycle": "bike",
    "car": "car", "motorcycle": "motorcycle", "skateboard": "skateboard",
    "surfboard": "surfboard", "airplane": "plane", "train": "train", "bus": "bus",
}
BG_TOPN = 4          # candidates sent to SAM2; masks_to_labelme keeps the best BG_MAX by area
BG_MIN_AREA = 1200
BG_DEDUP_IOU = 0.6
BG_PRE_SKIP_OVERLAP = 0.75   # only skip when the coarse mask is almost entirely on a person


def tight_box(mask):
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def main():
    coarse_prefix, out_prefix = sys.argv[1], sys.argv[2]
    extras_path = sys.argv[3] if len(sys.argv) > 3 else None

    masks = np.load(coarse_prefix + "_masks.npz")["masks"]
    inst = json.load(open(coarse_prefix + "_inst.json"))["instances"]
    H, W = masks.shape[1:]

    person = np.any([m for m, it in zip(masks, inst) if it["label"] == "person"], axis=0) \
        if any(it["label"] == "person" for it in inst) else np.zeros((H, W), bool)
    kern = np.ones((DILATE_PX * 2 + 1, DILATE_PX * 2 + 1), np.uint8)
    pdil = cv2.dilate(person.astype(np.uint8), kern, iterations=1).astype(bool)

    targets, refs, log = [], [], []
    for m, it in zip(masks, inst):
        if it["label"] == "person":
            tb = tight_box(m)
            targets.append({"name": "person", "box": tb, "src": "det",
                            "score": it["score"], "note": ""})
            refs.append(m)
        elif it["label"] in PORTABLE:
            x0, y0, x1, y1 = [int(round(v)) for v in it["box"]]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, x1), min(H, y1)
            box_area = max(1, (x1 - x0) * (y1 - y0))
            overlap = float(pdil[y0:y1, x0:x1].sum()) / box_area
            log.append(f'  {it["label"]:12s} {it["score"]:.2f} overlap={overlap:.2f} box={it["box"]}')
            if overlap >= OVERLAP_THR:
                targets.append({"name": PORTABLE[it["label"]], "box": tight_box(m), "src": "det",
                                "score": it["score"], "note": f'coco={it["label"]} overlap={overlap:.2f}'})
                refs.append(m)

    # drop duplicate detections of the same physical object: the detector sometimes emits
    # several boxes -- even under different COCO labels (e.g. phone + remote + knife on one
    # phone) -- for a single object. Compare the masks by IoU and keep the best-scoring one.
    pidx = [k for k, t in enumerate(targets) if t["src"] == "det" and t["name"] != "person"]
    drop = set()
    accepted = []
    for k in sorted(pidx, key=lambda k: (targets[k]["score"] or 0), reverse=True):
        for j in accepted:
            inter = float(np.logical_and(refs[k], refs[j]).sum())
            union = float(np.logical_or(refs[k], refs[j]).sum())
            if union > 0 and inter / union > 0.5:
                drop.add(k)
                break
        else:
            accepted.append(k)
    if drop:
        print(f"  - dropped {len(drop)} duplicate portable detection(s)")
        targets = [t for k, t in enumerate(targets) if k not in drop]
        refs = [r for k, r in enumerate(refs) if k not in drop]

    # --- prominent background entities (top-N by mask area, de-duplicated) ---
    bg_cands = []
    for m, it in zip(masks, inst):
        if it["label"] not in BG_NAME:
            continue
        area = int(m.sum())
        if area < BG_MIN_AREA:
            continue
        if person.any() and np.logical_and(m, person).sum() / area > BG_PRE_SKIP_OVERLAP:
            continue
        bg_cands.append((area, it["label"], m))
    bg_cands.sort(key=lambda c: -c[0])
    picked = []
    for area, lab, m in bg_cands:
        if any(np.logical_and(m, pm).sum() / np.logical_or(m, pm).sum() > BG_DEDUP_IOU for _, _, pm in picked):
            continue
        picked.append((area, lab, m))
        if len(picked) >= BG_TOPN:
            break
    for area, lab, m in picked:
        targets.append({"name": BG_NAME[lab], "box": tight_box(m), "src": "bg",
                        "score": None, "note": f'coco={lab} area={area}'})
        refs.append(m)
        print(f'  * bg {BG_NAME[lab]:10s} coco={lab:14s} area={area}')
    if len(picked) < BG_TOPN:
        print(f"  ! only {len(picked)} auto background candidate(s) (need {BG_TOPN})")

    if extras_path and os.path.exists(extras_path):
        data = json.load(open(extras_path))
        drops = {e["drop_coco"] for e in data if e.get("drop_coco")}
        drop_boxes = [e["drop_box"] for e in data if e.get("drop_box")]

        def in_any_box(t):
            cx = (t["box"][0] + t["box"][2]) / 2
            cy = (t["box"][1] + t["box"][3]) / 2
            return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in drop_boxes)

        keep = [k for k, t in enumerate(targets)
                if not any(d in t["note"] for d in drops) and not in_any_box(t)]
        if len(keep) != len(targets):
            print(f"  - dropped {len(targets) - len(keep)} false targets (coco={sorted(drops)}, boxes={len(drop_boxes)})")
        targets, refs = [targets[k] for k in keep], [refs[k] for k in keep]

        renames = {}
        for e in data:
            renames.update(e.get("rename_coco", {}))
        for t in targets:
            for coco, newname in renames.items():
                if f"coco={coco}" in t["note"]:
                    print(f'  ~ renamed {t["name"]} -> {newname} (coco={coco})')
                    t["name"] = newname
        for e in data:
            if not e.get("box"):
                continue
            x0, y0, x1, y1 = e["box"]
            ref = np.zeros((H, W), bool)
            ref[int(y0):int(y1), int(x0):int(x1)] = True
            targets.append({"name": e["label"], "box": [float(v) for v in e["box"]],
                            "src": "bg" if e.get("bg") else "manual",
                            "score": None, "note": e.get("note", "")})
            refs.append(ref)
            print(f'  + manual {e["label"]:12s} box={e["box"]}')

    print("portable candidates:")
    print("\n".join(log) if log else "  (none)")
    json.dump(targets, open(out_prefix + "_targets.json", "w"), indent=1)
    np.savez_compressed(out_prefix + "_ref.npz", refs=np.array(refs, dtype=bool) if refs else np.zeros((0, H, W), bool))
    names = [t["name"] for t in targets]
    print(f"targets={len(targets)}: {names}")


if __name__ == "__main__":
    main()
