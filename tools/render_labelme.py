"""Render a labelme json (filled polygons + labels) next to the image, for visual review.

Colours come from entities.color_for, i.e. they are keyed by LABEL NAME rather than by
shape index. That matters when rendering a sequence: the same entity keeps the same
colour in every frame, so a video of overlays is actually readable.

Usage: render_labelme.py <labelme.json> <image_path> <out_png>
"""
import json
import os
import sys

from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "pipeline"))
import entities as E  # noqa: E402 - needs the pipeline dir on sys.path first


def main():
    d = json.load(open(sys.argv[1]))
    im = Image.open(sys.argv[2]).convert("RGB")
    E.render_overlay(im, d["shapes"], sys.argv[3])
    print("saved", sys.argv[3], "shapes:", [s["label"] for s in d["shapes"]])


if __name__ == "__main__":
    main()
