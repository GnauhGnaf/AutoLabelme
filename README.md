# Auto-annotation pipeline

**English** | [中文](README.zh-CN.md)

Fully automatic instance segmentation: drop images into `input/`, run one
command, get [labelme](https://github.com/wkentaro/labelme) polygon JSON in
`output/`.

The pipeline is four stages — COCO Mask R-CNN for a coarse pass, a selection
step that keeps people + held objects + a couple of background entities, SAM2.1
(box-prompted, ONNX) for the refine, and a polygonizer that emits labelme JSON.

There is a second entry point for **video**: split the clip into frames, run the
same four stages on each, then keep every physical entity under ONE name for the
whole clip (`person1` is the same person in frame 0 and frame 123, and no entity
is ever given two names in one frame). See [Video](#video).

## Layout

```
.
  input/               INPUT  - the images to annotate  (<id>.<png|jpg|jpeg|bmp|webp|tif|tiff>)
  output/
    <id>.json          OUTPUT - generated labelme json
    ov/<id>.png        OUTPUT - overlay: masks drawn over the image
    <video>/           OUTPUT - a video run; see "Video" below
  run.sh               runner: input/ -> output/
  run_video.sh         runner: <video> -> output/<stem>/  (-> <stem>-s<N>/ when strided)
  deps/
    requirements-gpu.txt      env "gpu"  (Python 3.10) - all four stages
  pipeline/
    entities.py          shared entity layer: geometry, dedup, naming, labelme I/O,
                         overlay - used by BOTH the still and the video path
    coarse_seg.py        stage 1 - COCO Mask R-CNN coarse masks
    select_targets.py    stage 2 - pick persons + held objects + bg entities
    sam2_masks.py        stage 3 - SAM2.1 box-prompt refine (ONNX)
    masks_to_labelme.py  stage 4 - polygonize -> labelme json
    video_frames.py      video only - extract frames (cv2, no external ffmpeg)
    track.py             video only - cross-frame identity (one name per entity)
    video_annotate.py    video only - driver: frames -> stages -> tracker -> json
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
    verify_video.py      pure-code checks on a video run (exit 1 on failure)
  _out/                  intermediates (gitignored)
```

Image ids are the file stem, and the extension is free: `input/0486.png`,
`input/0486.jpg` and `input/0486.webp` all mean the id `0486` and all produce
`output/0486.json`. Two inputs sharing a stem is an error rather than a silent
overwrite.

Overlays live in a sibling `ov/` directory (`output/ov/0486.png`), not in an
`<id>_ov.png` next to the JSON.

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
bash run.sh 0486.jpg     # the extension is optional, and anything supported works
MODEL=sam2:small bash run.sh
```

Results go to `output/<id>.json`, each with a preview overlay `output/ov/<id>.png`
(masks drawn over the image). Intermediates stay in `_out/`.

Stage wiring, all paths relative to the repo root:

```
coarse_seg.py       input/<id>.<ext>  _out/<id>           0.25
select_targets.py   _out/<id> _out/<id>
sam2_masks.py       _out/<id>_targets.json _out/<id>_ref.npz input/<id>.<ext> _out/<id> sam2:latest 0.55
masks_to_labelme.py _out/<id>_masks.npz _out/<id>_masks.json input/<id>.<ext> output/<id>.json 1.0
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

## Video

```bash
bash run_video.sh pi-image-scene.mp4        # every frame
bash run_video.sh pi-image-scene.mp4 8      # every 8th frame (fast iteration)
STRIDE=8 bash run_video.sh a.mp4 b.mp4      # several clips, same stride
MODEL=sam2:small bash run_video.sh clip.mp4
```

An argument may be a path, a bare filename, or just the stem (looked up in
`input/`). With no arguments, every video sitting in the repo root is processed.

`pi-image-scene.mp4` above is the clip this was developed against; it is not in
the repo, since video files are gitignored. Point the command at your own.

Two requirements drive the design, and both are checked by code rather than by
eye:

1. **One entity, one name, for the whole clip.** `person1` in frame 0 is the same
   person in frame 123 — numbers never shuffle.
2. **No entity gets two names in one frame.** A person is never both `person1`
   and `person2`.

The still path satisfies neither: it numbers shapes per frame by centroid x
(`masks_to_labelme.py`, `entities.order_shapes` with `labels=None`), so the same
bag is `bag8` in one frame and `bag10` in the next.

### Output layout

```
output/<stem>[-s<N>]/
  frames/000000.json ...      one labelme 5.10.1 file per sampled frame
  ov/000000.png ...           overlay, colour keyed by LABEL NAME (stable across frames)
  tracks.json                 entity table + every identity decision
  preview.mp4                 the overlays, at the clip's real-time rate
_out/<stem>[-s<N>]/frames/*.png   the extracted frames (intermediates)
```

A stride other than 1 gets the `-s<N>` suffix (`output/pi-image-scene-s8/`), so a
strided pass and a full pass never share a directory — otherwise a 16-frame run
would silently truncate a 124-frame one.

Frame files are named by the **source** frame number, so under `--stride 8` they
are `000000.json`, `000008.json`, ..., and seeking in the clip works directly.
Extraction uses cv2 — no external `ffmpeg` binary is needed.

None of this is committed, and neither is the sample clip. Frame JSONs embed
their frame as base64, so a 124-frame clip is ~235 MB of output on top of a
video file that is itself large; `.gitignore` excludes video files and every
`output/<stem>/` directory, so the whole run is regenerated with `run_video.sh`.
The still path's `output/<id>.json` and `output/ov/` are unaffected — `ov/` is
negated back in.

`tracks.json` is the audit trail:

```json
{"label": "person3", "class": "person", "first_frame": 16, "last_frame": 123,
 "n_samples": 41, "gaps": [[40, 48]]}
```

`gaps` lists the source-frame ranges where a confirmed entity was not detected —
an occlusion, not a lost identity.

### How identity works

Per frame the pipeline runs the same four stages as the still path, then hands
the entities to a tracker (`pipeline/track.py`) before anything is written.

- **Association.** Tracks are kept per class and matched to each frame's
  detections by Hungarian assignment (`scipy.optimize.linear_sum_assignment`)
  over a cost mixing mask IoU, colour appearance, centroid distance and area
  ratio. The cost is what does the work where a greedy IoU match is not enough:
  two people walking past each other are a high-IoU ambiguity that only
  appearance breaks.
- **The gate is OR, not AND.** A pair is admissible when it clears *either* a
  mask-IoU bar or a centroid-distance bar. Requiring both would strand a
  fast-moving or briefly-occluded entity; accepting either still rejects "the
  person on the left is now the person on the right".
- **Names come last.** A track needs `MIN_HITS` (3) sightings before it earns a
  number, so a single-frame false positive can never burn one. Numbering is
  seeded by `(first frame, centroid x)` — which makes frame 0 come out
  left-to-right exactly as the still-image convention does — and frozen from
  there.
- **Re-identification.** A track unmatched for `LOST_MAX` (24) samples goes
  dormant rather than dying; a later detection whose appearance cosine clears
  `REID_SIM` (0.75) reclaims its number. Someone who walks out of shot and back
  in is still `person1`.
- **Post-hoc merge.** Two confirmed tracks that never co-occur and look alike
  are merged, on thresholds stricter than the per-frame re-ID. This covers two
  cases tracking cannot reach: a reappearance whose gap or position failed the
  association gate, and — the structurally impossible one — an object the
  *detector* labels `book` in one frame and `laptop` in the next, which becomes
  two entities because tracking matches within a class. Cross-class merging is
  refused outright when either side is a person.

Every merge, re-ID and drop is logged into `tracks.json`'s `events`, with its
evidence (appearance similarity, sample gap, pixel distance).

The thresholds are module constants at the top of `track.py`, and the ones that
matter are echoed into `tracks.json` under `params`, so a result is always
self-describing.

### Verifying a run

```bash
python tools/verify_video.py output/pi-image-scene
python tools/verify_video.py output/a --compare output/b   # + determinism
```

Pure code, no model and no agent in the loop. Exit status is 1 if any check
fails, so it can gate a run:

1. **structure** — every JSON is labelme 5.10.1, polygon shapes, ≥3 points, and
   the embedded image matches `imageWidth`/`imageHeight`.
2. **duplicates** — no two shapes in one frame share a label. This is
   requirement 2, checked mechanically.
3. **continuity** — an identity swap moves *two* entities at the same step, so
   that — not a single fast-moving object — is what it looks for.
4. **cross-name overlap** — two same-class entities under different names never
   overlap heavily; if they did, the within-frame dedup missed a duplicate.
   Non-person cross-class overlaps are reported but not failed: a legitimate
   nesting (a phone inside a bag) looks exactly like a duplicate from geometry
   alone, and the two cannot be told apart.
5. **table** — the entity table, with gaps and events.
6. **determinism** — with `--compare`, two runs at the same stride must produce
   identical frame JSONs and an identical entity table.

### Known limitations

These are detector false positives, not identity errors — verified by looking at
the overlays, and they are what the still path's `drop_coco` / `drop_box` hooks in
`extras.json` exist to clean up:

- `person10` in the sample clip is a luggage trolley plus a man's arm, which Mask
  R-CNN fires "person" on.
- `remote2` is a mis-detection on a person's arm.
- A bag strap is segmented as its own bag; it is disjoint from the real bag, so
  mask-IoU dedup cannot merge them.
- One spurious mask sits on a dark doorway region.


