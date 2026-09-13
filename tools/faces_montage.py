"""Build a face-strip montage for one image: crops the head region of every
person polygon in the labelme JSON, upscales each tile, lays them out in one
row with `${name}` labels.  One Read of the montage gives expression + race
for all persons at once.

Usage:  python faces_montage.py <id>            e.g. python faces_montage.py 0004
In:     output/<id>.json + input/<id>.png   (relative to the label/ dir)
Out:    tools/<id>_faces.png
"""
import json
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)  # the label/ directory


def head_box(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
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
    persons = [s for s in j['shapes'] if s['label'].startswith('person')]
    tiles = []
    TILE = 240
    for s in persons:
        x1, y1, x2, y2 = head_box(s['points'])
        c = img[y1:y2, x1:x2]
        if c.size == 0:
            continue
        s_ = TILE / c.shape[1]
        c = cv2.resize(c, (TILE, max(1, int(c.shape[0] * s_))),
                       interpolation=cv2.INTER_CUBIC)
        tile = np.full((TILE, TILE, 3), 255, np.uint8)
        tile[:min(c.shape[0], TILE)] = c[:TILE]
        cv2.rectangle(tile, (0, 0), (TILE - 1, TILE - 1), (0, 0, 0), 2)
        cv2.putText(tile, s['label'], (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        tiles.append(tile)
    if not tiles:
        print('no persons'); return
    mont = np.hstack(tiles)
    out = os.path.join(BASE, f'{iid}_faces.png')
    cv2.imwrite(out, mont)
    print(out, mont.shape, 'persons:', [s['label'] for s in persons])


if __name__ == '__main__':
    main()
