# 图像自动标注管线

[English](README.md) | **中文**

全自动实例分割：把图片放进 `input/`，跑一条命令，在 `output/` 得到
[labelme](https://github.com/wkentaro/labelme) 多边形 JSON。

管线共四个阶段：COCO Mask R-CNN 做粗检 → 一个筛选步骤保留「人物 + 手持物 +
少量背景实体」→ SAM2.1（box prompt，ONNX）精修 → 多边形化并输出 labelme JSON。

## 目录结构

```
.
  input/               输入 - 待标注图片  (<id>.png)
  output/              输出 - 生成的 labelme json  (<id>.json) + 叠加预览图 (<id>_ov.png)
  run.sh               运行脚本: input/ -> output/
  deps/
    requirements-gpu.txt      环境 "gpu"      (Python 3.10) - 阶段 1 和 4
    requirements-labelme.txt  环境 "labelme"  (Python 3.9)  - 阶段 3
  pipeline/
    coarse_seg.py        阶段 1 - COCO Mask R-CNN 粗掩码
    select_targets.py    阶段 2 - 选出人物 + 手持物 + 背景实体
    sam2_masks.py        阶段 3 - SAM2.1 box-prompt 精修 (ONNX)
    masks_to_labelme.py  阶段 4 - 多边形化 -> labelme json
  tools/                 复核结果时的可选辅助脚本
    render_labelme.py    labelme json -> 叠加图
    crop_grid.py         拼图式裁剪
    panel.py             单图总览：场景 + 人脸条 + shape 列表
    compare_gt.py        预测 vs 真值对比
    eval_iou.py          IoU 评分
    probe.py             单框临时探测
    exp_sam2.py          SAM2 prompt 配置实验
    patch_merge.py       合并 / 替换结果中的掩码
    faces_montage.py     人脸条拼图
  _out/                  中间产物（已 gitignore）
```

图片 id 即文件名主干：`input/0486.png` → `output/0486.json`。

## 安装

两个 conda 环境，各自一个 pin 文件：

```bash
conda create -n gpu python=3.10 -y && conda activate gpu
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r deps/requirements-gpu.txt

conda create -n labelme python=3.9 -y && conda activate labelme
pip install -r deps/requirements-labelme.txt
```

（把 `cu128` 换成你的 CUDA 版本，或改用 CPU wheel 源。阶段 1 在 CPU 上也能跑，
只是慢一些。）

`run.sh` 默认调用 `conda run -n gpu` / `conda run -n labelme`。如果你不用 conda，
可直接指定解释器：

```bash
GPU_PY=/path/to/gpu/python LME_PY=/path/to/labelme/python bash run.sh
```

全部为本地推理——无需 API key。唯一的联网行为是下面的一次性模型下载。

## SAM2 模型下载

阶段 3 通过 [`osam`](https://github.com/wkentaro/osam) 运行 **SAM2.1**，由该包把模型
以 ONNX 形式交给 onnxruntime 执行。权重**不**随代码分发；osam 会在首次使用时用
`gdown` 从该项目的 GitHub release 自动下载。

默认模型 `sam2:latest` 即 **SAM2.1 base_plus**，需要两个文件：

| 部件 | 文件 | 大小 | sha256 |
|------|------|------|--------|
| 编码器（图像预处理） | [`sam2.1_base_plus_preprocess.onnx`](https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_base_plus_preprocess.onnx) | 306 MB | `ce95c44082b4532c25ae01e11da3c9337dab7b04341455c09ae599dc9ae5c438` |
| 解码器 | [`sam2.1_base_plus.onnx`](https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_base_plus.onnx) | 21 MB | `2ad091af889b20ad2035503b4355cd8924fcf0e29fa6536924c48dc220ecdc56` |

可以提前拉取（需要联网），或者直接让第一次 `run.sh` 去下载：

```bash
conda run -n labelme python -m osam pull sam2:latest
conda run -n labelme python -m osam list      # 查看已缓存内容
```

同一 release 里的其它尺寸，用 `MODEL=... bash run.sh` 切换：

| 模型 | 文件 |
|------|------|
| `sam2:tiny` | `sam2.1_tiny_preprocess.onnx`, `sam2.1_tiny.onnx` |
| `sam2:small` | `sam2.1_small_preprocess.onnx`, `sam2.1_small.onnx` |
| `sam2:base_plus`（= `sam2:latest`） | `sam2.1_base_plus_*.onnx` |
| `sam2:large` | `sam2.1_large_preprocess.onnx`, `sam2.1_large.onnx` |

### 缓存位置与 Windows 注意事项

osam 把 blob 存在 `~/.cache/osam/models/blobs/<hash>`，其中 `<hash>` 形如
`sha256:<hexdigest>`。在 Windows 上，名字里的 `:` 会让它变成 **NTFS 备用数据流**：
真实文件是 `%USERPROFILE%\.cache\osam\models\blobs\sha256`，每个模型存在以十六进制
摘要命名的**流**里。`ls` 只会看到一个 0 字节文件——要用 `dir /r` 才能看到大小：

```
2026/09/12  16:28                 0 sha256
                20,642,497 sha256:2ad091af...dc56:$DATA
               306,084,742 sha256:ce95c440...c438:$DATA
```

因此，在 Windows 上手动下载 `.onnx` 丢进 blobs 目录**是无效的**。请用
`osam pull`（在一台有网络的机器上跑一次）；在 Linux/macOS 上缓存就是普通目录，
正常拷贝即可。

## 运行

```bash
bash run.sh              # 跑 input/ 里所有图片
bash run.sh 0486 0487    # 只跑指定 id
MODEL=sam2:small bash run.sh
```

结果输出到 `output/<id>.json`，并附带一张预览叠加图 `output/<id>_ov.png`
（把掩码画在图上）。中间产物留在 `_out/`。

各阶段调用方式（路径均相对于仓库根目录）：

```
coarse_seg.py       input/<id>.png  _out/<id>           0.25
select_targets.py   _out/<id> _out/<id>
sam2_masks.py       _out/<id>_targets.json _out/<id>_ref.npz input/<id>.png _out/<id> sam2:latest 0.55
masks_to_labelme.py _out/<id>_masks.npz _out/<id>_masks.json input/<id>.png output/<id>.json 1.0
```

## 标注内容

1. **人物** —— 所有检测到的人，标为 `person1..personN`，按多边形质心的 x
   从左到右编号。
2. **手持物** —— 粗掩码与人物轮廓 / 手部区域有重叠的物体（COCO 类别映射为短
   名称），同类内编号。
3. **背景实体** —— 至多 **2** 个显著的场景物体（椅子、货架、柜台、柜子……），
   按掩码面积保留；与人物重叠过多的背景掩码会被丢弃。

## 输出格式

labelme 5.10.1 JSON，`shape_type: "polygon"`，`imageData` 内嵌 base64 RGB。
shape 按类别名分组，并在类内按质心 x 升序编号。
