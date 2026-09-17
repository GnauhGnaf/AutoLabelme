# Auto-annotation pipeline

**English** | [中文](README.zh-CN.md)

Fully automatic instance segmentation: drop images into `input/`, run one
command, get [labelme](https://github.com/wkentaro/labelme) polygon JSON in
`output/`.

The pipeline is four stages — COCO Mask R-CNN for a coarse pass, a selection
step that keeps people + held objects + a couple of background entities, SAM2.1
(box-prompted, ONNX) for the refine, and a polygonizer that emits labelme JSON.

## Layout

```
.
  input/               INPUT  - the images to annotate  (<id>.png)
  output/              OUTPUT - generated labelme jsons  (<id>.json) + overlay (<id>_ov.png)
  run.sh               runner: input/ -> output/
  deps/
    requirements-gpu.txt      env "gpu"  (Python 3.10) - all four stages
  pipeline/
    coarse_seg.py        stage 1 - COCO Mask R-CNN coarse masks
    select_targets.py    stage 2 - pick persons + held objects + bg entities
    sam2_masks.py        stage 3 - SAM2.1 box-prompt refine (ONNX)
    masks_to_labelme.py  stage 4 - polygonize -> labelme json
  tools/                 optional helpers used while reviewing results
    render_labelme.py    labelme json -> overlay png
    crop_grid.py         contact-sheet crops
    panel.py             scene + face strip + shape list, one image
    compare_gt.py        compare prediction vs ground truth
    eval_iou.py          IoU scoring
    probe.py             ad-hoc single-box probe
    exp_sam2.py          SAM2 prompt-config experiments
    patch_merge.py       merge / replace masks in a result
    faces_montage.py     face-strip montage
  _out/                  intermediates (gitignored)
```

Image ids are the file stem: `input/0486.png` -> `output/0486.json`.

## Install

One conda env, `gpu`, covers all four stages:

```bash
conda create -n gpu python=3.10 -y && conda activate gpu
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r deps/requirements-gpu.txt
```

(Swap `cu128` for your CUDA version, or use the CPU wheel index. Stage 1 works
on CPU, just slower. Stage 3 runs SAM2.1 on CPU via onnxruntime either way.)

`run.sh` defaults to `conda run -n gpu`. If you don't use conda, point it at your
interpreter instead:

```bash
PY=/path/to/python bash run.sh
```

All inference is local — no API keys. The only network use is the one-time model
download below.

## SAM2 model download

Stage 3 runs **SAM2.1** through the [`osam`](https://github.com/wkentaro/osam)
package, which executes it as ONNX via onnxruntime. Weights are **not** vendored;
osam downloads them on first use (`gdown`) from the project's GitHub releases.

The default model `sam2:latest` is **SAM2.1 base_plus** — two files:

| part | file | size | sha256 |
|------|------|------|--------|
| encoder (image preprocess) | [`sam2.1_base_plus_preprocess.onnx`](https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_base_plus_preprocess.onnx) | 306 MB | `ce95c44082b4532c25ae01e11da3c9337dab7b04341455c09ae599dc9ae5c438` |
| decoder | [`sam2.1_base_plus.onnx`](https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_base_plus.onnx) | 21 MB | `2ad091af889b20ad2035503b4355cd8924fcf0e29fa6536924c48dc220ecdc56` |

Pull them up front (needs network) or just let the first `run.sh` fetch them:

```bash
conda run -n gpu python -m osam pull sam2:latest
conda run -n gpu python -m osam list      # what's cached
```

Other sizes from the same release, selected with `MODEL=... bash run.sh`:

| model | files |
|-------|-------|
| `sam2:tiny` | `sam2.1_tiny_preprocess.onnx`, `sam2.1_tiny.onnx` |
| `sam2:small` | `sam2.1_small_preprocess.onnx`, `sam2.1_small.onnx` |
| `sam2:base_plus` (= `sam2:latest`) | `sam2.1_base_plus_*.onnx` |
| `sam2:large` | `sam2.1_large_preprocess.onnx`, `sam2.1_large.onnx` |

### Cache location, and a Windows caveat

osam stores a blob at `~/.cache/osam/models/blobs/<hash>` where `<hash>` is
`sha256:<hexdigest>`. On Windows the `:` turns that name into an **NTFS alternate
data stream**: the real file is `%USERPROFILE%\.cache\osam\models\blobs\sha256`
and each model lives in a stream named after its digest. `ls` shows a single
0-byte file — use `dir /r` to see sizes:

```
2026/09/12  16:28                 0 sha256
                20,642,497 sha256:2ad091af...dc56:$DATA
               306,084,742 sha256:ce95c440...c438:$DATA
```

So dropping a manually-downloaded `.onnx` into the blobs folder does **not** work
on Windows. Use `osam pull` (once, on a networked machine); on Linux/macOS the
cache is a plain directory and copies normally.

## Run

```bash
bash run.sh              # every image in input/
bash run.sh 0486 0487    # only these ids
MODEL=sam2:small bash run.sh
```

Results go to `output/<id>.json`, each with a preview overlay `output/<id>_ov.png`
(masks drawn over the image). Intermediates stay in `_out/`.

Stage wiring, all paths relative to the repo root:

```
coarse_seg.py       input/<id>.png  _out/<id>           0.25
select_targets.py   _out/<id> _out/<id>
sam2_masks.py       _out/<id>_targets.json _out/<id>_ref.npz input/<id>.png _out/<id> sam2:latest 0.55
masks_to_labelme.py _out/<id>_masks.npz _out/<id>_masks.json input/<id>.png output/<id>.json 1.0
```

## What gets labelled

1. **People** — every detected person, as `person1..personN`, numbered
   left→right by polygon centroid x.
2. **Held objects** — objects whose coarse mask touches a person's silhouette /
   hand region (COCO classes mapped to short names), numbered within class.
3. **Background entities** — at most **2** prominent scene objects (chairs,
   shelves, counters, cabinets, ...) kept by mask area; a background mask that
   overlaps people too much is dropped.

## Output format

labelme 5.10.1 JSON, `shape_type: "polygon"`, `imageData` base64-embedded RGB.
Shapes are grouped by class name and numbered by centroid x ascending.
