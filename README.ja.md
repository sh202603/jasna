[English](README.md) | [**日本語**](README.ja.md) | [中文](README.zh.md)

# <img width="32" src="https://github.com/Kruk2/jasna/blob/main/assets/jasna-logo.png?raw=true" /> Jasna 

Jasna は、シンプルな GUI、CLI、GPU 専用処理パイプライン、NVIDIA TensorRT と実験的な AMD ROCm 対応、任意のセカンダリ復元モデル、静止画復元、ストリーミング機能を備えた JAV モザイク復元ツールです。

[Lada](https://codeberg.org/ladaapp/lada) に着想を得ており、一部は Lada をベースにしています。Jasna で使っている `mosaic_restoration_1.2` 復元モデルは、Lada 作者の ladaapp によって訓練されました。

Jasna は無料です。支援者には、このプロジェクト用に訓練された追加モデルを解除するキーが提供されます: **unet-4x** セカンダリアップスケーラーと、実験的な **SD 1.5 画像復元**モデルです。詳しくは[プロジェクトを支援する](#プロジェクトを支援する)をご覧ください。

> ### ⚙️ これは Jasna の `+modi` フォークです
>
> 上流 [Kruk2/jasna](https://github.com/Kruk2/jasna) をベースにした改変ビルドで、**フレーム生成**（`--frame-gen` 2x/4x）と**柔軟な出力**（HEVC/AV1, 8/10-bit, BT.601/709/2020 色空間）などを追加しています。
>
> - **ソース（このフォーク/ブランチ）:** [sh202603/jasna @ `modi`](https://github.com/sh202603/jasna/tree/modi)
> - **上流との変更点一覧:** [docs/CHANGES_vs_upstream_ja.md](docs/CHANGES_vs_upstream_ja.md)
> - **対象範囲 — 公開（無料）機能のみ。** 支援者モデル（**unet-4x** と **SD 1.5 画像復元**）は支援者キーで解錠される暗号化チェックポイントで、復号コードは**この公開フォークに含まれない**プライベートサブモジュールにあります。そのため、これらのモデルは**ここではダウンロード・復号・実行できません**。上流のコードは同梱されますが inert（不活性）のままです。支援者モデルが必要な場合は上流 [**Kruk2/jasna**](https://github.com/Kruk2/jasna) を使い支援者になってください。それ以外（検出・動画復元・RTX/TVAI セカンダリ・エクスポート後アクション・AV1・フレーム生成）は通常通り動作します。

<img width="1200" height="907" alt="image" src="https://github.com/user-attachments/assets/d59a914b-482d-4f37-ae72-5c59eb5dc9bb" />

## 目次

- [Jasna でできること](#jasna-でできること)
- [`+modi` の追加機能](#modi-の追加機能)
- [コミュニティ](#コミュニティ)
- [要件](#要件)
- [クイックスタート](#クイックスタート)
- [初回実行](#初回実行)
- [さらに詳しく](#さらに詳しく)
- [ベンチマーク](#ベンチマーク)
- [プロジェクトを支援する](#プロジェクトを支援する)
- [TODO](#todo)

## Jasna でできること

- 動画ファイルのモザイクを復元します。
- 実験的な SD 1.5 画像モデルで静止画のモザイクを復元します。
- 標準では RF-DETR モデルでモザイクを検出します。Lada と ZeLeFans の YOLO モデルも利用できます。
- サイドバイサイド VR180 動画を左右の目ごとに処理し、検出と復元にはオプションで魚眼再投影も使えます。
- テンポラルオーバーラップとクロスフェードで、クリップ境界のフリッカーを軽減します。
- オプションの[セカンダリ復元モデル](docs/ja/models.md) — **unet-4x**、**RTX Super Resolution**、**Topaz Video AI** — で品質をさらに高められます。復元した領域、とくに大きなモザイク、クローズアップ、4K 動画がシャープになります。
- 復元した動画を内蔵ブラウザプレーヤーや対応する Stash フォークへストリーミングできます。
- **`+modi`:** HEVC または AV1、8/10-bit、BT.601/709/2020 色空間を保持して出力できます。
- **`+modi`:** AI フレーム生成（RIFE）による 2x/4x のフレームレート アップコンバート。

## `+modi` の追加機能

これらの機能は `+modi` フォーク専用です。変更点の全一覧は [docs/CHANGES_vs_upstream_ja.md](docs/CHANGES_vs_upstream_ja.md) を参照してください。

### 柔軟な出力（コーデック / ビット深度 / 色空間）

これまで HEVC / 10-bit (P010) / BT.709 に固定されていた出力段が、**AV1**、**8-bit (NV12)** または **10-bit** に対応し、ソースの **BT.601 / BT.709 / BT.2020** 色空間を保持します。パイプラインは GPU 上で end-to-end のゼロコピーを維持します。

```bash
jasna --input input.mp4 --output output.mkv --codec av1 --bit-depth 10
```

`--codec {hevc,av1}`、`--bit-depth {auto,8,10}`。詳細: [docs/CODECS_AND_COLORSPACE_ja.md](docs/CODECS_AND_COLORSPACE_ja.md)。

### フレーム生成（フレームレート アップコンバート）

`--frame-gen {2x,4x}` は、ソースフレーム間に AI 補間フレーム（RIFE）を挿入して出力フレームレートを上げます。ファイル出力のみ（`--stream` では不可）。音声のタイムコードは維持されるため、長さと同期は保たれます。既定で fp16 動作 — fp32 比で約 1.9 倍高速（1080p 2x, RTX 5060 Ti）、見た目は同等です。

```bash
jasna --input input.mp4 --output output.mkv --frame-gen 2x
```

バックエンドは `--frame-gen-backend {rife,rtx}` で選択（`rife` が既定で現在利用可能。`rtx` は NVIDIA の `nvidia-vfx` リリース待ち）。

スタンドアロンの `jasna-framegen` コマンドは、**復元済み動画にフレーム生成だけ**を適用します（検出・復元なし）。2パス運用（先に復元、例えば公式バイナリで → 後からアップコンバート）に便利です。フォルダ入出力と `--output-pattern` 命名テンプレートにも対応（動画のみ）:

```bash
jasna-framegen --input restored.mkv --output out2x.mkv --factor 2x
jasna-framegen --input in_dir --output out_dir --factor 2x --output-pattern "{original}_2x.mkv"
```

詳細: [docs/FRAME_GENERATION_ja.md](docs/FRAME_GENERATION_ja.md)。

### 動画バックエンド（実験的）

Jasna は既定でデコードに `python_vali`、エンコードに `PyNvVideoCodec` を使います（`--video-backend native`）。**実験的**な [torchcodec](https://github.com/meta-pytorch/torchcodec) バックエンドは、8-bit HEVC/AV1 出力でこの両方を置き換えられます。

```bash
jasna --input input.mp4 --output output.mkv --video-backend auto
```

`--video-backend {native,auto,torchcodec}`（既定 `native`、つまり従来挙動）: `auto` は torchcodec が使える場面で使い、それ以外はネイティブにフォールバックします。`torchcodec` は強制します。`--decode-backend` / `--encode-backend` でデコード側・エンコード側を個別に上書きできます。torchcodec エンコーダは**8-bit HEVC/AV1**とマッピング可能な NVENC 設定に対応します。**10-bit、フレーム生成、ストリーミング、マッピング不可のエンコーダ設定は常にネイティブにフォールバック**し、色空間メタデータはどちらでも保持されます。オプション依存（cu130 wheel index から `pip install "torchcodec>=0.14.0"`）が必要です。詳細: [docs/TORCHCODEC_BACKEND_ja.md](docs/TORCHCODEC_BACKEND_ja.md)。

### FP8 復元バックエンド（実験的）

`--fp8-recon` は BasicVSR++ の upsample ステージを、TensorRT FP16 サブエンジンの代わりに cuDNN FP8 畳み込みで実行します。

```bash
jasna --input input.mp4 --output output.mp4 --fp8-recon
```

主な利点は VRAM です。TensorRT upsample エンジンがロード時に確保するアリーナ（既定 `--max-clip-size 90` で約 2.2GB）を確保しなくなり、480p〜4K のクリップで peak VRAM が 1.2〜1.7GB 下がることを実測しています。ステージ単体は約 1.5 倍速くなりますが、パイプラインの律速は検出側なので全体 fps は変わりません。出力は FP16 エンジンと目視で区別できず、走行間でビット決定的です。FP8 対応 GPU（sm89 以上、RTX 40 系以降。速度利得の実測は Blackwell のみ）と fp16 モードが必要で、失敗時は TensorRT エンジンへ自動フォールバックします。Linux と Windows の両方で動作確認済みです。詳細: [docs/FP8_RECON_ja.md](docs/FP8_RECON_ja.md)。

## コミュニティ

[SLS Discord](https://discord.gg/uNwQ4mHqgv) では、復元例、サポート、設定について話せます。あまり変な振る舞いはしないでください。

## 要件

- NVIDIA の **GTX 16 シリーズ / RTX 20 シリーズ以降**の GPU。GTX 10 シリーズ以前のカード（GTX 1050/1060/1070/1080）では動作しません。自分の GPU が対応か分からない場合は、NVIDIA の [GPU 一覧](https://developer.nvidia.com/cuda/gpus)で確認してください — コンピュート能力 7.5 以上が必要です。
- Nvidia ドライバーは Windows で **610 以上**、Linux で **580 以上**。
- AMD 対応は実験的で、ROCm 対応 GPU が必要です。
- Jasna は、パスに英語の文字と数字のみを含むフォルダにインストールしてください。

Jasna は VRAM を自動管理します。VRAM が不足すると、待機中のフレームを一時的にシステム RAM へ移します。設定は不要です。

## クイックスタート

1. OS と GPU ベンダーに合うリリースパッケージをダウンロードします。
2. パスに英語の文字のみを含むフォルダへ展開します。
3. アプリを起動します:
   - Windows: `jasna.exe` をダブルクリックします。
   - Linux NVIDIA: `jasna` ファイルを実行します。
   - Linux AMD: `run_jasna_amd.sh` を実行します。
4. 動画または画像を追加し、設定を選んで処理を開始します。

GUI のすべての設定にはツールチップがあります — 横の ⓘ アイコンにマウスを合わせてください。キューの並べ替え、プリセット、出力テンプレートなどの残りの機能は [GUI ガイド](docs/ja/gui.md)で紹介しています。

コマンドラインの方が好みですか？

```bash
# Single video
jasna --input input.mp4 --output output.mkv

# Still image
jasna --input photo.png --output restored.png

# Whole folder
jasna --input input_folder --output output_folder
```

`jasna --help` で全オプションを表示するか、[CLI リファレンス](docs/ja/cli.md)をご覧ください。

## 初回実行

初回実行は、お使いのカード専用の GPU ファイルを準備するため時間がかかります。NVIDIA では通常 **15～60 分**かかり、AMD では準備はずっと短時間です。これは一度だけで、結果は `model_weights` にキャッシュされ、以降の実行で再利用されます。古い Jasna バージョンから新しいバージョンへコピーすることもできます。

ブラウザを含む他のアプリを閉じ、実行中は PC の使用を避けてください。

処理中に VRAM が不足する場合は、まず**最大クリップサイズ**を下げてください。例: `180` から `60`。詳しくは [VRAM と GPU 使用量の調整](docs/ja/tuning.md)をご覧ください。

## さらに詳しく

- **[GUI の使い方](docs/ja/gui.md)** — キュー（ドラッグ＆ドロップ、並べ替え）、プリセット、出力テンプレートとファイル競合、その他の見落としがちな機能。
- **[モデルの選び方](docs/ja/models.md)** — 検出モデルの選択、セカンダリ復元（unet-4x / RTX Super Resolution / Topaz）によるよりシャープな結果、SD 1.5 静止画復元。
- **[動画の一部だけを復元する](docs/ja/segments.md)** — 区間エディター、内蔵モザイクスキャン、より良いマスクの提案、CLI の `--segments` フラグ。
- **[VR180 動画](docs/ja/vr180.md)** — Jasna がサイドバイサイド VR をどう扱うか、魚眼モードの使いどころ。
- **[VRAM と GPU 使用量の調整](docs/ja/tuning.md)** — クリップサイズ、テンポラルオーバーラップ、モデルコンパイル、VRAM 不足時の対処。
- **[高度な処理](docs/ja/advanced_processing.md)** — ノイズ除去、60→30 FPS 書き出し、カラー LUT、カスタムエンコーダー設定、エクスポート後のアクション。
- **[ストリーミング](docs/ja/streaming.md)** — 復元した動画をブラウザや Stash でリアルタイムに視聴。
- **[CLI リファレンス](docs/ja/cli.md)** — 出力テンプレート、コーデック別エンコーダー設定、エクスポート後のアクションを含む、すべてのコマンドラインオプション。
- **[ソースから実行](docs/en/development.md)** — 開発者向けセットアップとビルドメモ（英語）。

> **`+modi` ビルドガイド:** このフォークはネイティブ GPU ライブラリをビルドし、Jasna を**ソースから実行**します。公開のパッケージ済み/凍結バイナリはありません — パッケージングツールはプライベートな `jasna/protection` サブモジュールにあります（上流と同じ構成）。CUDA 13.0 ツールチェーン、ネイティブライブラリ、ffmpeg 8 / mkvmerge、TensorRT エンジン設定を網羅した手順:
> - Linux: [docs/BUILDING_LINUX_ja.md](docs/BUILDING_LINUX_ja.md)（[English](docs/BUILDING_LINUX_en.md)）
> - Windows: [docs/BUILDING_WINDOWS_ja.md](docs/BUILDING_WINDOWS_ja.md)（[English](docs/BUILDING_WINDOWS_en.md)）

## ベンチマーク

RTX 5090 + i9 13900k:

| ファイル                        | クリップ (秒) | lada 0.10.1 | jasna 0.3.0          | jasna 0.5.0          | **jasna 0.6.2**        |
| ------------------------------- | -------: | ----------: | --------------------:| --------------------:| ----------------------:|
| **ABF-017** (4k, 2時間25分)     | 60       | 02:56:26    | 01:20:49 (2.2倍高速) | 01:10:00 (2.5倍高速) | — |
| **HUBLK-063** (1080p, 3時間10分) | 180      | 01:34:51    | 44:21 (2.1倍高速)    | 37:57 (2.5倍高速)    | **30:58 (3.1倍高速)** |
| **DASS-570_2m**                 | 30       | 01:08       | 00:30 (2.3倍高速)    | 00:24 (2.8倍高速)    | **00:20 (3.4倍高速)** |
| **NASK-223_Test**               | 30       | 03:12       | 01:18 (2.5倍高速)    | 01:02 (3.1倍高速)    | **00:58 (3.3倍高速)** |
| **test-007**                    | 30       | 01:16       | 00:41 (1.9倍高速)    | 00:28 (2.7倍高速)    | **00:22 (3.5倍高速)** |
| **厚码测试2**                   | 30       | 01:52       | 00:43 (2.6倍高速)    | 00:36 (3.1倍高速)    | **00:34 (3.3倍高速)** |

## プロジェクトを支援する

支援は追加モデルの訓練に使われます。主に、GPU のレンタル代と、より大きなデータセットで訓練するための計算時間です。支援者には以下を解除するキーが提供されます:

- **unet-4x** セカンダリアップスケーラー。よりシャープな 256->1024 復元用です。
- **SD 1.5 画像復元**。実験的な静止画モデルです。

結果例:

- [unet-4x / セカンダリ復元の例（SLS Discord）](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260)
- [SD 1.5 画像復元の例（SLS Discord）](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) と [追加の SD 1.5 例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)

キーの入手方法:

1. 任意の回数、任意の時期で、合計 **15 USD 以上**を支援します。
2. 支援が処理されると、支援者キーが自動送信されます:
   - **[Unifans](https://app.unifans.io/c/kruk2)**: プラットフォームメッセージで送信されます。少し遅れる場合があります。
   - **[Buy Me a Coffee](https://buymeacoffee.com/kruk2)**、**暗号通貨**を含む: 支援時に使ったメールアドレスまたはハンドルへ送信されます。キーはそのメールアドレスまたはハンドルに紐付きます。

## TODO

現在の TODO:

- SeedVR 対応？
- パフォーマンスと VRAM の継続的な改善。
- より良い復元モデル。
- より良い検出モデル。
