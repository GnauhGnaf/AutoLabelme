"""Step B (gpu env): SAM2.1 box-prompt refine of the selected targets.

Best config from exp_sam2.py: box-only prompt using the TIGHT box, labels [2,3].
Guard: for detector targets, if SAM2 disagrees with the coarse ref mask (IoU < fallback_iou),
keep the ref mask instead (manual targets have a rectangle ref, so the guard is skipped).

  gpu env: python sam2_masks.py <targets.json> <ref.npz> <image> <out_prefix> [model] [fallback_iou]

The video driver imports run() instead of shelling out; get_model() keeps the ONNX
sessions alive across frames. Only the model is cached - encode_image() is inherently
per-frame, so the driver hands each frame's pixels in exactly once per call.
"""
import sys, os, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import entities as E  # noqa: E402 - same directory

from osam import apis, types

_MODEL = None
_MODEL_NAME = None


def get_model(model_name="sam2:latest"):
    """The osam SAM2 model, built at most once per process (rebuilt if the name changes).

    apis.get_model_type_by_name(...)() is what actually loads the ONNX sessions; that
    startup cost is why this is cached rather than constructed per frame.
    """
    global _MODEL, _MODEL_NAME
    if _MODEL is None or _MODEL_NAME != model_name:
        _MODEL = apis.get_model_type_by_name(model_name)()
        _MODEL_NAME = model_name
    return _MODEL


def run(targets, refs=None, img_path=None, out_prefix=None, model_name="sam2:latest",
        fallback_iou=0.55, model=None, arr=None, dump=True, verbose=True):
    """Box-prompt SAM2 on each target of one frame.

    ``targets`` is a list or a path to a targets JSON; ``refs`` is an (N, H, W) array, a
    path to a ref .npz, or None. ``arr`` optionally supplies the already-decoded RGB frame,
    so the driver does not re-open the image it just read. Returns {"masks", "meta"}.

    ``verbose`` off suppresses the one-line-per-target trace (124 frames x ~11 targets).
    """
    targets = E.read_json(targets)
    refs = None if refs is None else E.read_npz(refs, "refs")
    model = get_model(model_name) if model is None else model

    if arr is None:
        arr = np.array(Image.open(img_path).convert("RGB"))
    emb = model.encode_image(arr)

    meta, masks = [], []
    for k, t in enumerate(targets):
        x0, y0, x1, y1 = t["box"]
        prompt = types.Prompt(points=np.array([[x0, y0], [x1, y1]], dtype=np.float32),
                              point_labels=np.array([2, 3], dtype=np.int32))
        m = apis.generate(types.GenerateRequest(model=model_name, image_embedding=emb, prompt=prompt)).annotations[0].mask

        ref = refs[k] if refs is not None and k < len(refs) else None
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
        if verbose:
            suffix = "" if iou is None else f" iou_vs_ref={iou:.3f}"
            print(f'  {t["name"]:8s} src={t["src"]:6s} used={used}{suffix}')

    if dump and out_prefix:
        np.savez_compressed(out_prefix + "_masks.npz", masks=np.array(masks, dtype=bool))
        json.dump(meta, open(out_prefix + "_masks.json", "w"))
        print(f"saved {out_prefix}_masks.npz n={len(masks)} model={model_name}")
    else:
        print(f"{len(masks)} masks model={model_name}")
    return {"masks": masks, "meta": meta}


def main():
    targets_path, refs_path, img_path, prefix = sys.argv[1:5]
    model_name = sys.argv[5] if len(sys.argv) > 5 else "sam2:latest"
    fallback_iou = float(sys.argv[6]) if len(sys.argv) > 6 else 0.55
    run(targets_path, refs_path, img_path, prefix, model_name, fallback_iou)


if __name__ == "__main__":
    main()
