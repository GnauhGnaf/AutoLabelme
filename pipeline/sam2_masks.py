"""Step B (labelme env): SAM2.1 box-prompt refine of the selected targets.

Best config from exp_sam2.py: box-only prompt using the TIGHT box, labels [2,3].
Guard: for detector targets, if SAM2 disagrees with the coarse ref mask (IoU < fallback_iou),
keep the ref mask instead (manual targets have a rectangle ref, so the guard is skipped).

  labelme env: python sam2_masks.py <targets.json> <ref.npz> <image> <out_prefix> [model] [fallback_iou]
"""
import sys, json
import numpy as np
from PIL import Image

from osam import apis, types


def main():
    targets = json.load(open(sys.argv[1]))
    refs = np.load(sys.argv[2])["refs"]
    img_path, prefix = sys.argv[3], sys.argv[4]
    model_name = sys.argv[5] if len(sys.argv) > 5 else "sam2:latest"
    fallback_iou = float(sys.argv[6]) if len(sys.argv) > 6 else 0.55

    arr = np.array(Image.open(img_path).convert("RGB"))
    model = apis.get_model_type_by_name(model_name)()
    emb = model.encode_image(arr)

    meta, masks = [], []
    for k, t in enumerate(targets):
        x0, y0, x1, y1 = t["box"]
        prompt = types.Prompt(points=np.array([[x0, y0], [x1, y1]], dtype=np.float32),
                              point_labels=np.array([2, 3], dtype=np.int32))
        m = apis.generate(types.GenerateRequest(model=model_name, image_embedding=emb, prompt=prompt)).annotations[0].mask

        ref = refs[k] if k < len(refs) else None
        used = model_name
        iou = None
        if ref is not None and t["src"] == "det":
            u = np.logical_or(m, ref).sum()
            iou = float(np.logical_and(m, ref).sum() / u) if u else 0.0
            if iou < fallback_iou:
                m, used = ref, "coarse"
        masks.append(m)
        meta.append({"name": t["name"], "src": t["src"], "box": t["box"],
                     "sam_iou_vs_ref": None if iou is None else round(iou, 3), "used": used})
        suffix = "" if iou is None else f" iou_vs_ref={iou:.3f}"
        print(f'  {t["name"]:8s} src={t["src"]:6s} used={used}{suffix}')

    np.savez_compressed(prefix + "_masks.npz", masks=np.array(masks, dtype=bool))
    json.dump(meta, open(prefix + "_masks.json", "w"))
    print(f"saved {prefix}_masks.npz n={len(masks)} model={model_name}")


if __name__ == "__main__":
    main()
