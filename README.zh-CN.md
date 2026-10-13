# 图像自动标注管线

[English](README.md) | **中文**

全自动实例分割：把图片放进 `input/`，跑一条命令，在 `output/` 得到
[labelme](https://github.com/wkentaro/labelme) 多边形 JSON。

管线共四个阶段：COCO Mask R-CNN 做粗检 → 一个筛选步骤保留「人物 + 手持物 +
少量背景实体」→ SAM2.1（box prompt，ONNX）精修 → 多边形化并输出 labelme JSON。

另有一个**视频**入口：把片子切成帧、对每帧跑同样的四个阶段，然后让每个物理实体
在整段视频里**只对应一个名称**（`person1` 在第 0 帧和第 123 帧是同一个人，且任何一帧
内都不会有实体被标成两个名字）。见[视频](#视频)。

## 目录结构

```
.
  input/               输入 - 待标注图片  (<id>.<png|jpg|jpeg|bmp|webp|tif|tiff>)
  output/
    <id>.json          输出 - 生成的 labelme json
    ov/<id>.png        输出 - 叠加预览图：把掩码画在图上
    <video>/           输出 - 视频运行的产物，详见下文「视频」
  run.sh               运行脚本: input/ -> output/
  run_video.sh         运行脚本: <video> -> output/<stem>/（隔帧跑时为 <stem>-s<N>/）
  deps/
    requirements-gpu.txt      环境 "gpu"  (Python 3.10) - 全部四个阶段
  pipeline/
    entities.py          共享实体层：几何、去重、命名、labelme 读写、叠加渲染
                         —— 图片路径与视频路径共用
    coarse_seg.py        阶段 1 - COCO Mask R-CNN 粗掩码
    select_targets.py    阶段 2 - 选出人物 + 手持物 + 背景实体
    sam2_masks.py        阶段 3 - SAM2.1 box-prompt 精修 (ONNX)
    masks_to_labelme.py  阶段 4 - 多边形化 -> labelme json
    video_frames.py      仅视频 - 抽帧（cv2，不依赖外部 ffmpeg）
    track.py             仅视频 - 跨帧身份：一个实体一个名称
    video_annotate.py    仅视频 - 驱动器：帧 -> 各阶段 -> 跟踪 -> json
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
    verify_video.py      视频运行的纯代码自检（失败时退出码 1）
  _out/                  中间产物（已 gitignore）
```

图片 id 即文件名主干，扩展名不限：`input/0486.png`、`input/0486.jpg`、
`input/0486.webp` 都表示 id `0486`，都产出 `output/0486.json`。两个输入共用同一
主干会**直接报错**，而不是静默覆盖。

叠加预览图放在同级的 `ov/` 目录里（`output/ov/0486.png`），不再是 JSON 旁边的
`<id>_ov.png`。

## 安装

单个 conda 环境 `gpu` 即可跑全部四个阶段：

```bash
conda create -n gpu python=3.10 -y && conda activate gpu
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r deps/requirements-gpu.txt
```

（把 `cu128` 换成你的 CUDA 版本，或改用 CPU wheel 源。阶段 1 在 CPU 上也能跑，
只是慢一些；阶段 3 无论如何都是 onnxruntime 的 CPU 推理。）

`run.sh` 默认调用 `conda run -n gpu`。如果你不用 conda，可直接指定解释器：

```bash
PY=/path/to/python bash run.sh
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
conda run -n gpu python -m osam pull sam2:latest
conda run -n gpu python -m osam list      # 查看已缓存内容
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
bash run.sh 0486.jpg     # 扩展名可省，任何受支持的格式都行
MODEL=sam2:small bash run.sh
```

结果输出到 `output/<id>.json`，并附带一张预览叠加图 `output/ov/<id>.png`
（把掩码画在图上）。中间产物留在 `_out/`。

各阶段调用方式（路径均相对于仓库根目录）：

```
coarse_seg.py       input/<id>.<ext>  _out/<id>           0.25
select_targets.py   _out/<id> _out/<id>
sam2_masks.py       _out/<id>_targets.json _out/<id>_ref.npz input/<id>.<ext> _out/<id> sam2:latest 0.55
masks_to_labelme.py _out/<id>_masks.npz _out/<id>_masks.json input/<id>.<ext> output/<id>.json 1.0
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

## 视频

```bash
bash run_video.sh pi-image-scene.mp4        # 每一帧
bash run_video.sh pi-image-scene.mp4 8      # 每 8 帧取一帧（快速迭代）
STRIDE=8 bash run_video.sh a.mp4 b.mp4      # 多段片子，同一 stride
MODEL=sam2:small bash run_video.sh clip.mp4
```

参数可以是路径、纯文件名，或者只给主干名（会去 `input/` 里找）。不给参数时，
处理仓库根目录下所有视频。

整个设计由两条要求驱动，而且两条都由**代码**检查，不靠肉眼看：

1. **一个实体在整段视频里只有一个名称。** 第 0 帧的 `person1` 和第 123 帧的
   `person1` 是同一个人，序号不会跳变。
2. **任何一帧内，一个实体不会有两个名字。** 一个人不会既是 `person1` 又是
   `person2`。

图片路径两条都不满足：它每帧按质心 x 独立编号
（`masks_to_labelme.py` 里 `entities.order_shapes` 传 `labels=None`），
所以同一个包在这一帧是 `bag8`、下一帧变成 `bag10`。

### 产物布局

```
output/<stem>[-s<N>]/
  frames/000000.json ...      每个采样帧一个 labelme 5.10.1 文件
  ov/000000.png ...           叠加图，配色按「标签名」索引（跨帧稳定）
  tracks.json                 实体表 + 每一次身份判定
  preview.mp4                 叠加图串成的预览，按片子的真实时间速率播放
_out/<stem>[-s<N>]/frames/*.png   抽出的帧（中间产物）
```

stride 不为 1 时会带上 `-s<N>` 后缀（`output/pi-image-scene-s8/`），这样隔帧运行
与完整运行永远不会共用同一个目录——否则一次 16 帧的运行会把 124 帧的结果静默截断。

帧文件名用的是**源视频的帧号**，所以 `--stride 8` 时是 `000000.json`、
`000008.json`…… 直接就能对应到片子里去定位。抽帧走 cv2，**不需要**外部
`ffmpeg` 可执行文件。

帧 JSON 会把整帧以 base64 内嵌，所以 124 帧的产物约 235 MB；因此 `.gitignore`
只保留 `tracks.json`（体积小，且是每一次身份判定的凭据），`frames/`、`ov/` 与
`preview.mp4` 需要由 `run_video.sh` 重新生成。图片路径的 `output/ov/` 不受影响。

`tracks.json` 是可审计的凭据：

```json
{"label": "person3", "class": "person", "first_frame": 16, "last_frame": 123,
 "n_samples": 41, "gaps": [[40, 48]]}
```

`gaps` 列出某个已确认实体**没有被检测到**的源帧区间——那是一次遮挡，不是身份丢失。

### 身份是怎么维持的

每帧跑与图片路径完全相同的四个阶段，然后先把实体交给跟踪器
（`pipeline/track.py`），确认之后才写出任何东西。

- **关联**：轨迹按类别维护，与当帧检测用匈牙利算法
  （`scipy.optimize.linear_sum_assignment`）在代价矩阵上求最优指派，
  代价混合了掩码 IoU、颜色外观、质心距离和面积比。贪心的 IoU 匹配不够用的
  地方正是这个代价在起作用：两个人擦身而过是 IoU 上的高歧义，只有外观能拆开。
- **门限是「或」，不是「与」**：一对组合只要跨过掩码 IoU 门槛**或**质心距离门槛
  就可关联。要求同时满足会把快速移动或短暂被遮挡的实体甩掉；而只满足其一仍然
  足以否决「左边那个人现在变成右边那个人」。
- **名字最后才给**：轨迹要被看到 `MIN_HITS`（3）次才获得编号，因此单帧误检
  永远占不掉一个号。编号由 `(首次出现帧, 该帧内质心 x)` 播种——这正好让第 0 帧
  按从左到右输出、与图片路径的既有约定一致——此后冻结不变。
- **再识别**：连续 `LOST_MAX`（24）个采样帧未匹配的轨迹会转入休眠而不是死亡；
  之后若有检测的外观余弦超过 `REID_SIM`（0.75），它**认领回原编号**。
  走出画面再走回来的人，仍然是 `person1`。
- **后置合并**：两条从未同帧出现、外观又高度接近的已确认轨迹会被合并，阈值比
  逐帧再识别更严。这覆盖了跟踪本身够不到的两种情况：重现时的间隔或位置没过
  关联门限；以及结构上无解的一种——**检测器**这一帧叫 `book`、下一帧叫 `laptop`
  的同一个物体，会变成两个实体，因为跟踪是在类内匹配的。只要有一方是 person，
  跨类合并一律拒绝。

每一次合并、再识别、丢弃都会连同证据（外观相似度、采样间隔、像素距离）写进
`tracks.json` 的 `events`。

阈值都是 `track.py` 顶部的模块常量，其中关键几个会回写进 `tracks.json` 的
`params`，所以任何一次结果都是自描述的。

### 复核一次运行

```bash
python tools/verify_video.py output/pi-image-scene
python tools/verify_video.py output/a --compare output/b   # 追加确定性检查
```

纯代码，链路里没有模型也没有 agent。任何一项不通过则退出码为 1，可以直接用来
卡住一次运行：

1. **结构** —— 每帧 JSON 能按 labelme 5.10.1 解析，`shape_type: polygon`，
   点数 ≥3，内嵌图像与 `imageWidth`/`imageHeight` 一致。
2. **帧内无重复** —— 任一帧内不存在两个 shape 共用同一 label。这就是要求 2，
   由机器检查而非肉眼。
3. **身份连续性** —— 身份互换会让**两个**实体在同一步同时跳变，所以它找的是
   这个，而不是单个快速移动的物体。
4. **跨名重叠** —— 同类但不同名的两个实体不会大幅重叠；若重叠了，说明帧内去重
   漏掉了一个重复。非人物的跨类重叠只报告、不判失败：一个合法的嵌套（包里的
   手机）在纯几何上与重复长得一模一样，两者无法区分。
5. **实体表** —— 打印名称 / 类别 / 首末帧 / 出现帧数 / 断档，以及全部事件。
6. **确定性** —— 加 `--compare` 时，同一 stride 跑两次必须产出完全一致的各帧 JSON
   和实体表。

### 已知局限

以下都是**检测器误检**，不是身份错误——已通过查看叠加图确认。图片路径的
`extras.json` 里预留了 `drop_coco` / `drop_box` 正是用来清理这一类结果的：

- 示例片里的 `person10` 是一台行李车加一位男士的手臂，Mask R-CNN 在它上面
  误报了 "person"。
- `remote2` 是落在某人手臂上的误检。
- 一条包带被单独分割成了一个包；它与真正的包在掩码上不相交，所以基于掩码 IoU
  的去重无法把它们合并。
- 还有一个虚假掩码落在一处昏暗的门口区域上。
