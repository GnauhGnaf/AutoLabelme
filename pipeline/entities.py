"""Shared entity layer: mask geometry, within-frame dedup, naming, labelme I/O, overlay.

Used by both the still-image path (masks_to_labelme.py) and the video path
(video_annotate.py / track.py) so the two cannot drift apart.

Everything here is deterministic: pure numpy/PIL, no model, and deliberately no
``hash()`` (Python salts string hashes per process, which would make overlay colours
- and therefore the run itself - irreproducible).
"""
import base64
import colorsys
import json
import os
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- within-frame duplicate elimination --------------------------------------
DEDUP_IOU_SAME = 0.55       # same class: masks overlapping this much are one object
DEDUP_CONTAIN_SAME = 0.85   # same class: one mask swallowing another (a bag and its handle)
DEDUP_IOU_CROSS = 0.80      # different class: only a near-total overlap is a duplicate.
                            # Deliberately IoU and not containment - a book held in a hand
                            # sits inside the person's mask (high containment, low IoU) and
                            # must survive as its own shape.
HUE_STEP = 0.11             # hue offset between successive indices of one class
FONT_SIZE = 26


# --- input discovery ---------------------------------------------------------
# Keep in sync with IMAGE_EXTS in run.sh and run_video.sh
IMAGE_EXTS = ("png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff")


def find_input(root, iid):
    """Path of ``input/<iid>.<ext>`` for any supported extension.

    Mirrors run.sh's id resolution so the tools and the runner agree on what an id means.
    Raises FileNotFoundError when nothing matches, ValueError when several extensions of
    the same stem exist (both would write the same output/<id>.json).
    """
    d = os.path.join(root, "input")
    hits = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        stem, dot, ext = name.rpartition(".")
        if dot and stem == iid and ext.lower() in IMAGE_EXTS:
            hits.append(p)
    if not hits:
        raise FileNotFoundError(f"no input image for id {iid!r} in {d}")
    if len(hits) > 1:
        raise ValueError(f"id {iid!r} is ambiguous, several inputs share that stem: {hits}")
    return hits[0]


def load_bgr(path):
    """Read an image as BGR uint8, i.e. what cv2 consumers expect - but via PIL.

    cv2.imread() fails on non-ASCII paths on Windows (it goes through the ANSI code page),
    so tools that only want a numpy array should come through here instead.
    """
    return np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def read_json(src):
    """An already-loaded object, or a path to a JSON file holding one.

    Lets each stage's run() accept either its CLI input (a path) or the previous stage's
    in-memory result (an object), so the video driver can chain stages without a single
    intermediate file and still be exercising the exact code path the CLI does.
    """
    return json.load(open(src)) if isinstance(src, (str, bytes, os.PathLike)) else src


def read_npz(src, key):
    """An already-loaded array, or a path to an .npz holding it under ``key``."""
    return np.load(src)[key] if isinstance(src, (str, bytes, os.PathLike)) else src


# --- geometry ----------------------------------------------------------------
def rasterize(points, W, H):
    """Polygon -> bool mask (H, W). Lifted from tools/eval_iou.py:11."""
    m = Image.new("1", (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in points], fill=1)
    return np.array(m, dtype=bool)


def iou(a, b):
    """Binary mask IoU, with a zero-union guard. Lifted from tools/exp_sam2.py:42."""
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def containment(a, b):
    """|a & b| / min(|a|, |b|) - 1.0 when the smaller mask sits entirely inside the other."""
    m = min(a.sum(), b.sum())
    return float(np.logical_and(a, b).sum() / m) if m else 0.0


def centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.mean(xs)), float(np.mean(ys))


# --- within-frame dedup ------------------------------------------------------
def _rank(item):
    """'Which duplicate survives': detector/SAM2 score first, then mask area.

    Returns a tuple so it can be used directly as a ``max`` key.
    """
    s = item.get("score")
    return (float(s) if s is not None else 0.0, int(item["mask"].sum()))


def dedup(items):
    """Collapse duplicates inside ONE frame.

    ``items`` is a list of dicts with at least {"name", "mask", "score"}. Two items are
    the same physical object when they overlap enough (see the DEDUP_* thresholds above).
    Duplicates are union-found, so 3+ copies of one object collapse to a single survivor.

    ``mask`` is expected to be the geometry that will actually be delivered - i.e. a
    polygon rasterized back, not a raw network mask. Polygonizing keeps only the largest
    contour and so changes how much two entities overlap; judging duplicates on the raw
    masks let one bag through twice. Callers get this right by polygonizing first:
    see masks_to_labelme.build_items.

    Returns ``(keep_indices, dropped)`` where ``dropped`` is a list of
    ``(dropped_index, kept_index, reason)`` suitable for logging. Nothing is ever dropped
    silently.
    """
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            ov = iou(a["mask"], b["mask"])
            if a["name"] == b["name"]:
                # Low index wins the root, which keeps grouping deterministic.
                if ov > DEDUP_IOU_SAME or containment(a["mask"], b["mask"]) > DEDUP_CONTAIN_SAME:
                    ra, rb = find(i), find(j)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
            elif ov > DEDUP_IOU_CROSS:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    keep, dropped = [], []
    for root in sorted(groups):
        idxs = groups[root]
        if len(idxs) == 1:
            keep.append(idxs[0])
            continue
        best = idxs[0]
        for i in idxs[1:]:
            if _rank(items[i]) > _rank(items[best]):
                best = i
        keep.append(best)
        for i in idxs:
            if i != best:
                dropped.append((i, best, f'{items[i]["name"]} overlaps {items[best]["name"]}'))
    keep.sort()
    return keep, dropped


# --- naming ------------------------------------------------------------------
def split_label(label):
    """'person12' -> ('person', 12). A label with no numeric suffix gets index 1."""
    i = len(label)
    while i > 0 and label[i - 1].isdigit():
        i -= 1
    if i == len(label):
        return label, 1
    n = int(label[i:])
    return label[:i], (n if n > 0 else 1)


def order_shapes(items, labels=None):
    """Order items canonically and attach labels.

    Order is 'person' first, then the remaining classes alphabetically, and within a
    class by centroid x ascending - the convention the still-image pipeline already used
    (masks_to_labelme.py:76-81).

    ``labels`` maps an index into ``items`` to its label. When omitted, labels are
    '<class><n>' numbered in that same order (the still-image convention). The video path
    passes labels from the tracker instead, so identity wins over geometry; the ordering
    then only decides the order shapes appear in the JSON, not what they are called.
    """
    order = []
    for cls in sorted({it["name"] for it in items}, key=lambda c: (c != "person", c)):
        group = [(i, it) for i, it in enumerate(items) if it["name"] == cls]
        group.sort(key=lambda pair: centroid(pair[1]["points"])[0])
        order.extend(group)

    shapes, counters = [], {}
    for i, it in order:
        if labels is None:
            counters[it["name"]] = counters.get(it["name"], 0) + 1
            label = f'{it["name"]}{counters[it["name"]]}'
        else:
            label = labels[i]
        shapes.append({"label": label, "points": it["points"]})
    return shapes


# --- rendering ---------------------------------------------------------------
def _stable_hue(name):
    """Stable hash of a class name -> [0, 1). crc32, NOT Python's salted hash()."""
    return (zlib.crc32(name.encode("utf-8")) % 360) / 360.0


def hue_for(label):
    """Deterministic colour keyed by the LABEL NAME, not by shape index.

    tools/render_labelme.py:18 used ``np.linspace`` over the shape index, so the same
    entity changed colour whenever the shape order shifted - fatal across video frames.
    Here the hue is a pure function of the label, so person1 keeps one colour for the
    whole clip.
    """
    cls, n = split_label(label)
    return (_stable_hue(cls) + HUE_STEP * (n - 1)) % 1.0


def color_for(label):
    return tuple(int(v * 255) for v in colorsys.hsv_to_rgb(hue_for(label), 0.78, 0.95))


def render_overlay(im, shapes, out_path):
    """Filled polygons + label chips over a copy of ``im``. Ported from
    masks_to_labelme.py:93-111, with the palette re-keyed by label name."""
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)

    ov = im.convert("RGBA")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    for s in shapes:
        c = color_for(s["label"])
        dr.polygon([tuple(p) for p in s["points"]], fill=c + (105,), outline=c + (255,))
    ov = Image.alpha_composite(ov, lay)

    dr = ImageDraw.Draw(ov)
    try:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()
    for s in shapes:
        p = np.array(s["points"], float)
        cx, cy = p[:, 0].mean(), p[:, 1].min()
        c = color_for(s["label"])
        dr.rectangle([cx - 2, cy - 30, cx + 8 + 13 * len(s["label"]), cy - 2],
                     fill=c + (235,))
        dr.text((cx + 2, cy - 29), s["label"], fill=(0, 0, 0, 255), font=font)
    ov.convert("RGB").save(out_path)
    return out_path


# --- labelme I/O -------------------------------------------------------------
def ov_path(out_json):
    """output/<id>.json -> output/ov/<id>.png. Previews live in a sibling ``ov/``
    directory rather than carrying an ``_ov`` suffix.

    Replaces the old ``out_path.replace(".json", "_ov.png")``, which also mangled any id
    that itself contained '.json'.

    Note ``ov/`` is a sibling of the JSON, so for a nested layout it lands inside the
    JSON's own directory. The video path wants ov/ beside frames/ and builds that path
    itself rather than going through here.
    """
    d, name = os.path.split(out_json)
    stem = name[:-5] if name.endswith(".json") else name
    return os.path.join(d, "ov", stem + ".png")


def write_labelme(im, img_path, shapes, out_path):
    """labelme 5.10.1 JSON with base64-embedded image bytes.

    ``imageData`` embeds the source file verbatim, so it stays whatever format the input
    was (labelme decodes it with PIL, which handles every format this pipeline accepts).
    """
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    W, H = im.size
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    json.dump({
        "version": "5.10.1", "flags": {},
        "shapes": [{"label": s["label"],
                    "points": [[round(x, 1), round(y, 1)] for x, y in s["points"]],
                    "group_id": None, "description": "", "shape_type": "polygon",
                    "flags": {}, "mask": None} for s in shapes],
        "imagePath": os.path.basename(img_path),
        "imageData": b64, "imageHeight": H, "imageWidth": W,
    }, open(out_path, "w"), indent=2)
    return out_path
