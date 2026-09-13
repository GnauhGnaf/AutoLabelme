import sys, numpy as np
from osam import apis, types
from PIL import Image

# usage: probe.py <img> "<label> x0 y0 x1 y1" ...
img = sys.argv[1]
arr = np.array(Image.open(img).convert('RGB'))
model = apis.get_model_type_by_name('sam2:latest')()
emb = model.encode_image(arr)

for spec in sys.argv[2:]:
    parts = spec.split()
    label = parts[0]
    x0, y0, x1, y1 = [float(v) for v in parts[1:5]]
    p = types.Prompt(points=np.array([[x0, y0], [x1, y1]], dtype=np.float32),
                     point_labels=np.array([2, 3], dtype=np.int32))
    m = np.array(apis.generate(types.GenerateRequest(
        model='sam2:latest', image_embedding=emb, prompt=p)).annotations[0].mask).astype(bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        print(f"{label}: EMPTY")
        continue
    print(f"{label}: area={int(m.sum())} x={xs.min()}-{xs.max()} y={ys.min()}-{ys.max()}")
    a = arr.copy()
    a[m] = (0.30 * a[m] + 0.70 * np.array([255, 0, 255])).astype('uint8')
    xa, xb, ya, yb = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    pad = 30
    crop = a[max(0, ya - pad):yb + pad, max(0, xa - pad):xb + pad]
    Image.fromarray(crop).save(f'_out/probe_{label}.png')
