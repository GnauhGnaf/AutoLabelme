"""One-read panel for an image: full scene on top, face strip at the bottom,
plus a printed shape list (labels + bboxes + centroid) so object ids can be
assigned without a second parse call.

Usage:  python panel.py <id>        e.g. python panel.py 0012
In:     output/<id>.json + input/<id>.png   (relative to the label/ dir)
Out:    tools/<id>_panel.png
"""
import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # the label/ directory

W = 1440
TILE = 240


def head_box(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    pad = 0.15 * w
    hh = min(0.28 * h, 1.2 * w)
    return (int(max(x1 - pad, 0)), int(max(y1 - pad, 0)),
            int(min(x2 + pad, 1e9)), int(min(y1 + hh, 1e9)))


def main():
    iid = sys.argv[1]
    j = json.load(open(os.path.join(ROOT, 'output', f'{iid}.json'),
                       encoding='utf-8'))
    img = cv2.imread(os.path.join(ROOT, 'input', f'{iid}.png'))
    H, Wimg = img.shape[:2]

    for s in j['shapes']:
        xs = [p[0] for p in s['points']]
        ys = [p[1] for p in s['points']]
        print(f"{s['label']}@({min(xs):.0f},{min(ys):.0f},{max(xs):.0f},{max(ys):.0f}) "
              f"c=({sum(xs)/len(xs):.0f},{sum(ys)/len(ys):.0f})")

    sc = W / Wimg
    top = cv2.resize(img, (W, int(H * sc)), interpolation=cv2.INTER_CUBIC)

    persons = [s for s in j['shapes'] if s['label'].startswith('person')]
    tiles = []
    for s in persons:
        x1, y1, x2, y2 = head_box(s['points'])
        c = img[y1:y2, x1:x2]
        if c.size == 0:
            continue
        c = cv2.resize(c, (TILE, max(1, int(c.shape[0] * TILE / c.shape[1]))),
                       interpolation=cv2.INTER_CUBIC)
        tile = np.full((TILE, TILE, 3), 255, np.uint8)
        tile[:min(c.shape[0], TILE)] = c[:TILE]
        cv2.rectangle(tile, (0, 0), (TILE - 1, TILE - 1), (0, 0, 0), 2)
        cv2.putText(tile, s['label'], (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        tiles.append(tile)
    bot = np.hstack(tiles) if tiles else np.full((TILE, W, 3), 255, np.uint8)
    if bot.shape[1] < W:
        pad = np.full((bot.shape[0], W - bot.shape[1], 3), 255, np.uint8)
        bot = np.hstack([bot, pad])
    else:
        bot = cv2.resize(bot, (W, int(bot.shape[0] * W / bot.shape[1])))

    out = np.vstack([top, np.full((6, W, 3), 0, np.uint8), bot])
    p = os.path.join(BASE, f'{iid}_panel.png')
    cv2.imwrite(p, out)
    print(p, out.shape)


if __name__ == '__main__':
    main()
