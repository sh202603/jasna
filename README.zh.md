[English](README.md) | [日本語](README.ja.md) | [**中文**](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna

Jasna 是一个 JAV 马赛克修复工具，提供简洁 GUI、CLI、纯 GPU 处理流水线、NVIDIA TensorRT 与实验性 AMD ROCm 支持、可选二级修复模型、静态图像修复以及流媒体支持。

Jasna 是免费的。支持者会获得一个密钥，用于解锁为本项目训练的额外模型: **unet-4x** 二级放大模型，以及实验性的 **SD 1.5 图像修复**模型。详情见[支持本项目](#支持本项目)。

> ### ⚙️ 这是 Jasna 的 `+modi` 分支
>
> 基于上游 [Kruk2/jasna](https://github.com/Kruk2/jasna) v0.8.1 的改版构建，新增**帧生成**（`--frame-gen` 2x/4x）、**torchcodec 视频后端**、**FP8 修复后端**、**FlashVSR 二级修复**等改进。
>
> - **源码（本分支）:** [sh202603/jasna @ `modi`](https://github.com/sh202603/jasna/tree/modi)
> - **与上游的完整变更:** [docs/en/changes_vs_upstream.md](docs/en/changes_vs_upstream.md)
> - **范围 — 仅公开（免费）功能。** 支持者模型（**unet-4x** 和 **SD 1.5 图像修复**）以加密检查点形式提供，需用支持者密钥解锁，而解密代码位于**不属于本公开分支**的私有子模块中 —— 因此这些模型**无法在此下载、解密或运行**。它们的上游代码随附但保持 inert（不激活）。如需支持者模型，请使用上游 [**Kruk2/jasna**](https://github.com/Kruk2/jasna) 并成为支持者。其他一切（检测、视频修复、RTX/TVAI 二级修复、区间编辑器、VR180、导出后操作、帧生成）均正常工作。

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />

## 目录

- [Jasna 能做什么](#jasna-能做什么)
- [`+modi` 新增功能](#modi-新增功能)
- [社区](#社区)
- [要求](#要求)
- [快速开始](#快速开始)
- [首次运行](#首次运行)
- [了解更多](#了解更多)
- [基准测试](#基准测试)
- [支持本项目](#支持本项目)
- [致谢](#致谢)
- [TODO](#todo)

## Jasna 能做什么

- 修复视频文件中的马赛克。
- 使用实验性 SD 1.5 图像模型修复静态图像中的马赛克。
- 默认使用快速的 `rfdetr-v6` 模型检测马赛克；也提供更大的 RF-DETR 变体，以及 Lada 和 ZeLeFans YOLO 模型。
- 可逐眼处理并排 VR180 视频，并按片商自动选择最合适的马赛克修复方式；还可在区间编辑器中预览每种方式。
- NVIDIA 和 AMD GPU 均支持按帧选择范围，区间编辑器还提供修复预览以及缩放和平移检查。
- 通过时间重叠和交叉淡化减少片段边界闪烁。
- 检测硬切镜头并在切换点结束跟踪片段，确保修复不会混合切换前后的画面。
- 可使用可选的[二级修复模型](docs/zh/models.md) — **unet-4x**、**RTX Super Resolution** 或 **Topaz Video AI** — 进一步提升质量，让修复区域更清晰，尤其是大面积马赛克、特写和 4K 视频。
- 内置原生 GUI 视频播放器，无需生成输出文件即可全屏播放和跳转修复画面。
- 可将修复后的视频串流到内置浏览器播放器，或支持的 Stash 分支。
- **`+modi`:** 通过 AI 帧生成（RIFE）实现 2x/4x 帧率提升。

## `+modi` 新增功能

这些功能为 `+modi` 分支专有。完整变更列表见 [docs/en/changes_vs_upstream.md](docs/en/changes_vs_upstream.md)。

### 帧生成（帧率提升）

`--frame-gen {2x,4x}` 通过在源帧之间插入 AI 插值帧（RIFE）来提升输出帧率。仅文件输出（不支持 `--stream`）；音频时间码保持不变，因此时长与同步得以保留。默认以 fp16 运行 —— 实测比 fp32 快约 1.9 倍（1080p 2x, RTX 5060 Ti），视觉效果一致。

```bash
jasna --input input.mp4 --output output.mkv --frame-gen 2x
```

后端通过 `--frame-gen-backend {rife,rtx}` 选择（`rife` 为默认且当前可用；`rtx` 等待 NVIDIA 的 `nvidia-vfx` 发布）。详情: [docs/en/frame_generation.md](docs/en/frame_generation.md)。

### 视频后端（实验性）

Jasna 默认通过 PyAV（NVDEC/NVENC）解码与编码（`--video-backend native`）。可选用一个**实验性**的 [torchcodec](https://github.com/meta-pytorch/torchcodec) 后端作为替代：

```bash
jasna --input input.mp4 --output output.mkv --video-backend auto
```

`--video-backend {native,auto,torchcodec}`（默认 `native`，即原有行为）：`auto` 在可用时将 torchcodec 用于**解码**，否则回退到 native；`torchcodec` 强制使用。`--decode-backend` / `--encode-backend` 可分别覆盖解码侧与编码侧。自 v0.8.0 起，native 编码器对 HEVC/AV1 始终输出 10-bit，而 torchcodec 的 NVENC 仅支持 8-bit，两者输出无法一致，因此 **torchcodec 编码仅在强制指定（`--encode-backend torchcodec`）时运行**，且仅限 8-bit 源与可映射的 NVENC 设置；流式传输、`--segments`、`--retarget-high-fps` 与帧生成保持 native。色彩空间元数据在两种情况下都会保留。需要可选依赖（从 cu130 wheel index 执行 `pip install "torchcodec>=0.15.0"`）。详情: [docs/en/torchcodec_backend.md](docs/en/torchcodec_backend.md)。

### FP8 修复后端（实验性）

`--fp8-recon` 用 cuDNN FP8 卷积代替 TensorRT FP16 子引擎来运行 BasicVSR++ 的 upsample 阶段：

```bash
jasna --input input.mp4 --output output.mp4 --fp8-recon
```

主要收益在 VRAM：不再分配 TensorRT upsample 引擎加载时的内部工作区（默认 `--max-clip-size 90` 下约 2.2GB），实测 480p 至 4K 片源的 VRAM 峰值降低 1.2–1.7GB。该阶段本身快约 1.5 倍，但流水线瓶颈在检测侧，整体 fps 不变。输出与 FP16 引擎在视觉上无法区分，且多次运行按位一致。需要支持 FP8 的 GPU（sm89 及以上，即 RTX 40 系或更新；加速仅在 Blackwell 上验证）和 fp16 模式；任何失败都会自动回退到 TensorRT 引擎。已在 Linux 和 Windows 上验证。详情: [docs/en/fp8_recon.md](docs/en/fp8_recon.md)。

### FlashVSR 二级修复（实验性）

`--secondary-restoration flashvsr` 用 [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) 扩散 VSR 将修复后的 256px 马赛克裁剪块放大到 1024px（4x），补回一级模型在大面积马赛克、特写和 4K 上留下的模糊纹理:

```bash
jasna --input in.mp4 --output out.mkv --secondary-restoration flashvsr --flashvsr-repo ~/FlashVSR_plus
```

FlashVSR 自身峰值 12–16GB VRAM，无法与约 9GB 的一级流水线共存。它以**峰值 VRAM 在时间上互不重叠的三个子进程**运行:(1) 一级修复 → 将裁剪块序列化到磁盘 *bundle*，(2) 在其专用 venv 中执行 FlashVSR 4x，(3) 重新混合并编码最终输出。你需自备 `FlashVSR_plus` 检出、其 v1.1 权重以及一个 **uv 托管的独立 Python venv**（系统 Python 无法 JIT 编译 FlashVSR 的 Triton 注意力内核）。仅文件输出；不兼容 `--stream` / `--frame-gen`。单趟版本 `--secondary-restoration flashvsr-inline` 在流水线内运行 FlashVSR，**无中间文件**（需 16GB 显卡以及打了 tiny-long 多块修复补丁的检出）。详情: [docs/en/flashvsr.md](docs/en/flashvsr.md)。

## 社区

加入 [SLS Discord](https://discord.gg/uNwQ4mHqgv) 查看示例、获取支持，并讨论设置。请不要表现得太奇怪。

## 要求

- NVIDIA **GTX 16 系列 / RTX 20 系列或更新**的 GPU。GTX 10 系列及更旧的显卡（GTX 1050/1060/1070/1080）无法使用。不确定自己的显卡？请查看 NVIDIA 的 [GPU 表格](https://developer.nvidia.com/cuda/gpus) — 需要计算能力 7.5 或更高。
- Nvidia 驱动: Windows 需 **610 或更新**，Linux 需 **580 或更新**。
- AMD 支持是实验性的，需要 ROCm 支持的 GPU。
- 请把 Jasna 安装到路径只包含英文字母和数字的文件夹中。

Jasna 会自动管理 VRAM: 显存不足时，等待中的帧会临时移动到系统内存。无需任何配置。

## 快速开始

1. 下载与你的操作系统和 GPU 厂商匹配的发行包。
2. 解压到路径只包含英文字符的文件夹。
3. 启动应用:
   - Windows: 双击 `jasna.exe`。
   - Linux NVIDIA: 运行 `jasna` 文件。
   - Linux AMD: 运行 `run_jasna_amd.sh`。
4. 添加视频或图像，选择设置，然后开始处理。

GUI 中的每个设置都有提示 — 把鼠标悬停在旁边的 ⓘ 图标上即可查看。[GUI 指南](docs/zh/gui.md)会带你了解其余功能: 重新运行整个队列、队列排序、媒体操作和播放器快捷方式、预设、输出文件名模板等等。

更喜欢命令行？

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image
jasna --input photo.png --output restored.png

# Whole folder
jasna --input input_folder --output output_folder
```

运行 `jasna --help` 查看全部选项，或阅读 [CLI 参考](docs/zh/cli.md)。

## 首次运行

首次运行会比较慢，因为 Jasna 需要针对你的显卡准备 GPU 专用文件。NVIDIA 通常需要 **15-60 分钟**；AMD 的准备时间要短得多。这只会发生一次 — 结果会缓存在 `model_weights` 中，之后每次运行都会复用。你也可以把它们从旧版本 Jasna 复制到新版本中。

请关闭其他应用程序（包括浏览器），并在此期间避免使用电脑。

如果处理时显存不足，请先降低**最大片段大小**，例如从 `180` 降到 `60`。见[调整 VRAM 和 GPU 占用](docs/zh/tuning.md)。

## 了解更多

- **[使用 GUI](docs/zh/gui.md)** — 修复视频播放器、队列、预设、输出文件名模板与文件冲突，以及其他容易错过的功能。
- **[选择模型](docs/zh/models.md)** — 该选哪个检测模型、用二级修复（unet-4x / RTX Super Resolution / Topaz）获得更清晰的结果，以及 SD 1.5 静态图像修复。
- **[只修复视频的一部分](docs/zh/segments.md)** — 区间编辑器、内置马赛克扫描、检测/修复模型的 A/B 对比、提交更好的遮罩，以及 `--segments` CLI 参数。
- **[VR180 视频](docs/zh/vr180.md)** — Jasna 如何处理并排 VR，以及按片商的自动设置。
- **[调整 VRAM 和 GPU 占用](docs/zh/tuning.md)** — 片段大小、时间重叠、模型编译，以及显存不足时该怎么办。
- **[高级处理](docs/zh/advanced_processing.md)** — 降噪、60→30 FPS 导出、色彩 LUT、锐化、原样传给编码器的 CQ 质量设置、自定义编码器设置，以及队列级或逐视频导出后操作。
- **[流媒体](docs/zh/streaming.md)** — 在浏览器中或通过 Stash 实时观看修复后的视频。
- **[CLI 参考](docs/zh/cli.md)** — 所有命令行选项，包括 `--cq`、输出模板、各编解码器的编码器设置和导出后操作。
- **[从源代码运行](docs/en/development.md)** — 开发者环境搭建和构建说明。

> **`+modi` 构建指南:** 本分支构建原生 GPU 库并**从源码运行** Jasna。没有公开的打包/冻结二进制 —— 打包工具位于私有的 `jasna/protection` 子模块中（与上游相同）。涵盖 CUDA 13.0 工具链、原生库、ffmpeg 8 / mkvmerge 和 TensorRT 引擎设置的分步指南:
> - Linux: [docs/en/building_linux.md](docs/en/building_linux.md)（[日本語](docs/ja/building_linux.md)）
> - Windows: [docs/en/building_windows.md](docs/en/building_windows.md)（[日本語](docs/ja/building_windows.md)）

`+modi` 功能指南（英文/日文）:

- **[帧生成](docs/en/frame_generation.md)** — RIFE 2x/4x 帧率提升与独立工具 `jasna-framegen`。
- **[视频后端](docs/en/torchcodec_backend.md)** — 实验性 torchcodec 解码/编码后端。
- **[FP8 修复后端](docs/en/fp8_recon.md)** — cuDNN FP8 上采样阶段，降低峰值显存。
- **[FlashVSR 二级修复](docs/en/flashvsr.md)** — 离线/内联扩散 4x 放大。
- **[冻结构建](docs/en/frozen_build.md)** — 实验性 Nuitka 独立构建。
- **[与上游的完整变更](docs/en/changes_vs_upstream.md)** — 本分支的全部差异。

## 基准测试

在 Linux、RTX 5090（驱动 595.84）和 i9-13900K 上进行端到端修复。每个构建均使用
默认模型以及 `--max-clip-size 180 --temporal-overlap 15
--secondary-restoration none`。Lada Flatpak 0.11.0 使用高精度 `v2`
检测器、CUDA FP16、`--max-clip-length 180` 和 NVIDIA HEVC HQ
预设。冻结版和 Lada 的计时为舍弃预热运行后的一次实测；v0.9.0 和 v0.9.1 的
计时为预热后交替目标顺序运行三次的中位数。括号内的倍数以 Lada 为基准；由于
Lada 未运行 8K，该行改用 v0.9.0 作为基准。每个 SONE 片段有 6,056 帧（3:22），
8K VR 片段有 900 帧（15 秒）。

表中只列出速度或显存确实发生变化的版本，以及每种分辨率一个输入。全部版本和
编码见[完整基准测试报告](benchmarks/2026-07-26_release_matrix.md)。

| 输入 | Lada 0.11.0 v2 | v0.4.1 | v0.5.0 | v0.9.0 (4a171c9) | **v0.9.1 (18add8a)** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p H.264 8-bit | 01:47 (基准) | 01:35 (快 1.1 倍) | 00:45 (快 2.4 倍) | 00:34 (快 3.1 倍) | **00:32 (快 3.4 倍)** |
| 1080p H.264 8-bit | 02:02 (基准) | 01:46 (快 1.2 倍) | 00:47 (快 2.6 倍) | 00:39 (快 3.2 倍) | **00:34 (快 3.6 倍)** |
| 2160p H.264 8-bit | 04:56 (基准) | 03:23 (快 1.5 倍) | 01:22 (快 3.6 倍) | 01:15 (快 3.9 倍) | **01:03 (快 4.7 倍)** |
| 8K VR HEVC 8-bit 60 fps | — | — | — | 00:34 (8K 基准) | — |

GPU 显存为基准测试目标的中位数/峰值（GiB）；`—` 表示未运行。括号内的倍数以
中位数与 v0.4.1 比较；v0.4.1 未运行的 8K 行改用 v0.9.0 作为基准。Lada 不参与
该比较：它的显存占用低于任何 Jasna 版本。

| 输入 | Lada | v0.4.1 | v0.5.0 | v0.9.0 | **v0.9.1** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p H.264 8-bit | 1.8/3.1 | 17.6/19.9 (基准) | 8.6/9.0 (少 2.0 倍) | 8.6/9.1 (少 2.0 倍) | **4.2/4.6 (少 4.2 倍)** |
| 1080p H.264 8-bit | 2.0/3.3 | 18.9/21.4 (基准) | 9.5/10.2 (少 2.0 倍) | 9.3/10.1 (少 2.0 倍) | **4.8/5.6 (少 3.9 倍)** |
| 2160p H.264 8-bit | 2.6/4.1 | 26.0/30.5 (基准) | 12.9/15.0 (少 2.0 倍) | 11.4/15.4 (少 2.3 倍) | **7.2/10.9 (少 3.6 倍)** |
| 8K VR HEVC 8-bit 60 fps | — | — | — | 17.1/18.3 (8K 基准) | — |

v0.9.1 显存下降的原因是修复子引擎改为按固定批大小构建，而不是按片段大小构建，
详见[引擎批大小报告](benchmarks/2026-07-28_engine_batch_vram.md)。
RAM 的中位数/峰值、测试方法、原始数据及性能提交对比，请参阅
[完整基准测试报告](benchmarks/2026-07-26_release_matrix.md)。

### 旧版完整视频基准测试

下面保留原有的 Linux 测试结果，并补充新的 v0.7.2 和 v0.9.0 结果。测试使用
相同的 RTX 5090 和 i9-13900K，并为每个输入保留表中所列的片段大小。

| 文件 | 片段（秒） | Lada 0.10.1 | Jasna 0.3.0 | Jasna 0.5.0 | Jasna 0.6.2 | Jasna 0.7.2 | **Jasna 0.9.0 (7d9cc8c)** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ABF-017** (4K，2小时25分) | 60 | 02:56:26 | 01:20:49 (快 2.2 倍) | 01:10:00 (快 2.5 倍) | — | — | **42:17 (快 4.2 倍)** |
| **HUBLK-063** (1080p，3小时10分) | 180 | 01:34:51 | 44:21 (快 2.1 倍) | 37:57 (快 2.5 倍) | 30:58 (快 3.1 倍) | 23:38 (快 4.0 倍) | **18:01 (快 5.3 倍)** |
| **DASS-570_2m** | 30 | 01:08 | 00:30 (快 2.3 倍) | 00:24 (快 2.8 倍) | 00:20 (快 3.4 倍) | 01:05 (快 1.0 倍) | **00:22 (快 3.1 倍)** |
| **NASK-223_Test** | 30 | 03:12 | 01:18 (快 2.5 倍) | 01:02 (快 3.1 倍) | 00:58 (快 3.3 倍) | 01:07 (快 2.9 倍) | **01:01 (快 3.1 倍)** |
| **test-007** | 30 | 01:16 | 00:41 (快 1.9 倍) | 00:28 (快 2.7 倍) | 00:22 (快 3.5 倍) | 00:27 (快 2.8 倍) | **00:27 (快 2.8 倍)** |

## 支持本项目

支持用于训练额外模型，主要是租用 GPU，以及在更大数据集上训练所需的计算时间。支持者会获得一个密钥，用于解锁:

- **unet-4x** 二级放大模型，用于更清晰的 256->1024 修复。
- **SD 1.5 图像修复**，实验性静态图像模型。

结果示例:

- [unet-4x / 二级修复示例（SLS Discord）](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)
- [SD 1.5 图像修复示例（SLS Discord）](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) 和 [更多 SD 1.5 示例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)

如何获取密钥:

1. 累计贡献 **15 美元或以上**，不限次数、不限时间。
2. 贡献处理完成后，支持者密钥会自动发送:
   - **[Unifans](https://app.unifans.io/c/kruk2)**: 通过平台消息发送，可能会有轻微延迟。
   - **[Buy Me a Coffee](https://buymeacoffee.com/kruk2)**，包括**加密货币**: 发送到贡献时使用的邮箱或账号。密钥与该邮箱或账号绑定。

## 致谢

- **[Lada](https://codeberg.org/ladaapp/lada)（Codeberg）** — Jasna 受 Lada 启发，
  部分代码也基于 Lada。Jasna 使用的 `mosaic_restoration_1.2` 修复模型由 Lada 作者
  ladaapp 训练。
- **ZeLeFans** — 感谢提供 VR 检测模型，以及对马赛克形状和 VR 投影的深入分析。

## TODO

当前 TODO:

- SeedVR 支持？
- 持续改善性能和 VRAM 使用。
- 更好的修复模型。
- 更好的检测模型。
