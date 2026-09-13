"""Draw GT (red) vs predicted (green) polygons for visual diff.
Usage: compare_gt.py <pred.json> <gt.json> <image> <out_png> [x0 y0 x1 y1]
"""
import sys, json
from PIL import Image, ImageDraw

pred = json.load(open(sys.argv[1]))
gt = json.load(open(sys.argv[2]))
im = Image.open(sys.argv[3]).convert("RGB")
out = sys.argv[4]

# match pred<->gt by label-number order per type (best-effort for display)
dr = ImageDraw.Draw(im)


def draw(data, color, w):
    for s in data["shapes"]:
        pts = [tuple(p) for p in s["points"]]
        dr.line(pts + [pts[0]], fill=color, width=w)


draw(gt, (255, 0, 0), 3)
draw(pred, (0, 255, 0), 2)
if len(sys.argv) > 8:
    x0, y0, x1, y1 = (int(v) for v in sys.argv[5:9])
    im = im.crop((x0, y0, x1, y1))
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
im.save(out)
print("saved", out, "red=GT green=pred")
