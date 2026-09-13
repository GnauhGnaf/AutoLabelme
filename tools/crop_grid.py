"""Crop + upscale an image with a coordinate grid, for my own visual inspection.
Coordinates printed are in ORIGINAL image pixel space."""
import sys, os
from PIL import Image, ImageDraw, ImageFont

def main():
    path = sys.argv[1]
    x0, y0, x1, y1 = (int(v) for v in sys.argv[2:6])
    scale = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
    out = sys.argv[7] if len(sys.argv) > 7 else "_vision_label/_view.png"
    step = int(sys.argv[8]) if len(sys.argv) > 8 else 50

    im = Image.open(path).convert("RGB")
    W, H = im.size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = im.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    nw, nh = int(cw * scale), int(ch * scale)
    crop = crop.resize((nw, nh), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    # vertical lines at original-x multiples of step
    gx = (x0 // step) * step
    while gx <= x1:
        if gx >= x0:
            px = (gx - x0) * scale
            major = (gx % (step * 2) == 0)
            d.line([(px, 0), (px, nh)], fill=(255, 0, 0) if major else (255, 160, 160), width=1)
            if major:
                d.text((px + 2, 2), str(gx), fill=(255, 0, 0), font=font)
        gx += step
    gy = (y0 // step) * step
    while gy <= y1:
        if gy >= y0:
            py = (gy - y0) * scale
            major = (gy % (step * 2) == 0)
            d.line([(0, py), (nw, py)], fill=(0, 0, 255) if major else (160, 160, 255), width=1)
            if major:
                d.text((2, py + 2), str(gy), fill=(0, 0, 255), font=font)
        gy += step
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    crop.save(out)
    print(f"saved {out}  crop=({x0},{y0},{x1},{y1}) scale={scale} size={crop.size} step={step}")

if __name__ == "__main__":
    main()
