import sys, json
import numpy as np, cv2

# usage: patch_merge.py <id> <label> <kh> [kw]
# kh = vertical closing kernel height (bridges vertical gaps), kw = horizontal (defaults to kh)
img_id = sys.argv[1]
label = sys.argv[2]
kh = int(sys.argv[3]) if len(sys.argv) > 3 else 5
kw = int(sys.argv[4]) if len(sys.argv) > 4 else kh
npz = f'_out/{img_id}_masks.npz'
out = f'output/{img_id}.json'

masks = np.load(npz)['masks']
d = json.load(open(out))

# find target shape
shapes = d['shapes']
tgt = None
for s in shapes:
    if s['label'] == label:
        tgt = s
        break
if tgt is None:
    print('label not found', label); sys.exit(1)

H, W = masks[0].shape[:2]

# rasterise the target polygon and match the mask with the highest pixel IoU
ref = np.zeros((H, W), np.uint8)
cv2.fillPoly(ref, [np.array(tgt['points'], np.int32).reshape(-1, 1, 2)], 1)
best = None
for i, mk in enumerate(masks):
    mk = (mk > 0).astype(np.uint8)
    inter = int((mk & ref).sum()); uni = int((mk | ref).sum())
    iou = inter / max(uni, 1)
    if best is None or iou > best[0]:
        best = (iou, i, mk)

iou, i, mk = best
dil = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((kh, 1), np.uint8))
dil = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, np.ones((1, kw), np.uint8))
cs, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cs = [c for c in cs if cv2.contourArea(c) > 200]
c = max(cs, key=cv2.contourArea)
eps = 1.0
approx = cv2.approxPolyDP(c, eps, True)
pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
print(f'matched mask idx {i} iou={iou:.2f} close={kh}x{kw} pts {len(tgt["points"])} -> {len(pts)} ext', cv2.boundingRect(c))
tgt['points'] = pts
json.dump(d, open(out, 'w'))
print('wrote', out)
