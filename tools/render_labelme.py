"""Render a labelme json (filled polygons + labels) next to the image, for visual review.
Usage: render_labelme.py <labelme.json> <image_path> <out_png>
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAL = {}


def color(lbl):
    return PAL[lbl]


def main():
    d = json.load(open(sys.argv[1]))
    im = Image.open(sys.argv[2]).convert("RGB")
    hues = np.linspace(0, 1, len(d["shapes"]), endpoint=False)
    import colorsys
    for i, s in enumerate(d["shapes"]):
        PAL[s["label"]] = tuple(int(v * 255) for v in colorsys.hsv_to_rgb(hues[i], 0.75, 0.95))
    ov = im.convert("RGBA")
    lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(lay)
    for s in d["shapes"]:
        c = color(s["label"])
        dr.polygon([tuple(p) for p in s["points"]], fill=c + (110,), outline=c + (255,))
    ov = Image.alpha_composite(ov, lay)
    dr = ImageDraw.Draw(ov)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    for s in d["shapes"]:
        p = np.array(s["points"], dtype=float)
        cx, cy = p[:, 0].mean(), p[:, 1].min()
        c = color(s["label"])
        dr.rectangle([cx - 2, cy - 30, cx + 8 + 13 * len(s["label"]), cy - 2], fill=c + (235,))
        dr.text((cx + 2, cy - 29), s["label"], fill=(0, 0, 0, 255), font=font)
    ov.convert("RGB").save(sys.argv[3])
    print("saved", sys.argv[3], "shapes:", [s["label"] for s in d["shapes"]])


if __name__ == "__main__":
    main()
