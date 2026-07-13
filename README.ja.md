[English](README.md) | [**日本語**](README.ja.md) | [中文](README.zh.md)

# Jasna

Jasna は、シンプルな GUI、CLI、GPU 専用処理パイプライン、TensorRT 対応、任意のセカンダリ復元モデル、静止画復元、ストリーミング機能を備えた JAV モザイク復元ツールです。

[Lada](https://codeberg.org/ladaapp/lada) に着想を得ており、一部は Lada をベースにしています。Jasna で使っている `mosaic_restoration_1.2` 復元モデルは、Lada 作者の ladaapp によって訓練されました。

Jasna は無料です。支援者には、このプロジェクト用に訓練された追加モデルを解除するキーが提供されます: **unet-4x** セカンダリアップスケーラーと、実験的な **SD 1.5 画像復元**モデルです。詳しくは[プロジェクトを支援する](#プロジェクトを支援する)をご覧ください。

> ### ⚙️ これは Jasna の `+modi` フォークです
>
> 上流 [Kruk2/jasna](https://github.com/Kruk2/jasna) をベースにした改変ビルドで、**フレーム生成**（`--frame-gen` 2x/4x）と**柔軟な出力**（HEVC/AV1, 8/10-bit, BT.601/709/2020 色空間）などを追加しています。
>
> - **ソース（このフォーク/ブランチ）:** [sh202603/jasna @ `modi`](https://github.com/sh202603/jasna/tree/modi)
> - **上流との変更点一覧:** [docs/CHANGES_vs_upstream_ja.md](docs/CHANGES_vs_upstream_ja.md)
> - **対象範囲 — 公開（無料）機能のみ。** 支援者モデル（**unet-4x** と **SD 1.5 画像復元**）は支援者キーで解錠される暗号化チェックポイントで、復号コードは**この公開フォークに含まれない**プライベートサブモジュールにあります。そのため、これらのモデルは**ここではダウンロード・復号・実行できません**。上流のコードは同梱されますが inert（不活性）のままです。支援者モデルが必要な場合は上流 [**Kruk2/jasna**](https://github.com/Kruk2/jasna) を使い支援者になってください。それ以外（検出・動画復元・RTX/TVAI セカンダリ・エクスポート後アクション・AV1・フレーム生成）は通常通り動作します。

![Jasna GUI](https://github.com/user-attachments/assets/ae5d9b73-ea22-4263-8203-0ff89bbbcc51)

## 目次

- [Jasna でできること](#jasna-でできること)
- [`+modi` の追加機能](#modi-の追加機能)
- [コミュニティ](#コミュニティ)
- [要件](#要件)
- [クイックスタート](#クイックスタート)
- [エクスポート後のアクション](#エクスポート後のアクション)
- [初回実行](#初回実行)
- [モデルの選び方](#モデルの選び方)
- [品質と VRAM の調整](#品質と-vram-の調整)
- [ストリーミング](#ストリーミング)
- [ベンチマーク](#ベンチマーク)
- [プロジェクトを支援する](#プロジェクトを支援する)
- [現在の制限と TODO](#現在の制限と-todo)
- [ソースから実行](#ソースから実行)

## Jasna でできること

- 動画ファイルのモザイクを復元します。
- 実験的な SD 1.5 画像モデルで静止画のモザイクを復元します。
- 標準では RF-DETR モデルでモザイクを検出します。Lada YOLO モデルも利用できます。
- テンポラルオーバーラップとクロスフェードで、クリップ境界のフリッカーを軽減します。
- **unet-4x**、**RTX Super Resolution**、**Topaz Video AI** によるセカンダリ復元を使えます。
- 内蔵ブラウザプレーヤー、または対応する Stash フォークへ復元動画をストリーミングできます。
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

### FlashVSR セカンダリ復元（オフライン、実験的）

`--secondary-restoration flashvsr` は、復元された 256px のモザイククロップを [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) 拡散 VSR で 1024px（4x）へ拡大し、大きなモザイク・接写・4K で一次モデルがぼやけさせるテクスチャを補います:

```bash
jasna --input in.mp4 --output out.mkv --secondary-restoration flashvsr --flashvsr-repo ~/FlashVSR_plus
```

FlashVSR は単体で 12〜16GB VRAM を消費するため、一次パイプライン（約 9GB）と同時常駐できません。**ピーク VRAM が時間的に重ならない 3 つのサブプロセス**として動きます:(1) 一次復元 → クロップをディスクの *bundle* へ直列化、(2) 専用 venv で FlashVSR 4x、(3) 最終出力を再 blend + encode。`FlashVSR_plus` の checkout・v1.1 重み・**uv-managed の standalone Python venv**（system Python では FlashVSR の Triton アテンションカーネルを JIT できない）は利用者が用意します。ファイル出力専用で、`--stream` / `--frame-gen` とは併用不可。単一パス版 `--secondary-restoration flashvsr-inline` は、FlashVSR をストリーミングパイプラインに挟んで**中間ファイル無し**で実行します（16GB カード + tiny-long マルチチャンク修正パッチを当てた checkout が前提）。詳細: [docs/FLASHVSR_ja.md](docs/FLASHVSR_ja.md)。

## コミュニティ

[SLS Discord](https://discord.gg/5R2Rx5nBH) では、復元例、サポート、設定について話せます。あまり変な振る舞いはしないでください。

## 要件

- コンピュート能力 **7.5 以上**の新しい Nvidia GPU。
- GPU の目安: **GTX 16 シリーズ**、**RTX 20 シリーズ**、**RTX 30 シリーズ**、**RTX 40 シリーズ**、**RTX 50 シリーズ**、および新しいワークステーション/データセンター向けカード。
- 古すぎるもの: **GTX 10 シリーズ**。GTX 1050/1060/1070/1080 は対象外です。
- 正確な GPU 確認には NVIDIA の [CUDA GPU compute capability table](https://developer.nvidia.com/cuda/gpus) を参照してください。
- 最新の Nvidia ドライバー。テスト済みドライバー: **591.67**。59x 系が最低想定ドライバーです。
- インストール先パスは ASCII 文字のみである必要があります。
- Windows リリースパッケージ: `ffmpeg`、`ffprobe`、`mkvmerge` を同梱しています。
- Linux リリースパッケージ: システム側に `ffmpeg`、`ffprobe`、`mkvmerge` が必要です。`ffmpeg` のメジャーバージョンは **8** である必要があります。`mkvmerge` は [MKVToolNix](https://mkvtoolnix.download/downloads.html) に含まれます。

Jasna は VRAM を自動管理します。GPU VRAM が不足すると、処理キューで待機中のフレームを一時的にシステム RAM へ移し、必要なときに戻します。設定は不要です。

## クイックスタート

1. 最新の Windows または Linux リリースパッケージをダウンロードします。
2. ASCII 文字のみのパスに展開します。
3. アプリを起動します:
   - Windows: `jasna.exe` をダブルクリックします。
   - Linux: `jasna` ファイルを実行します。
4. GUI に動画または画像を追加し、設定を選んで処理を開始します。

コマンドラインからも使えます:

```bash
jasna --input input.mp4 --output output.mkv
```

静止画では、画像専用フラグは不要です:

```bash
jasna --input photo.png --output restored.png
```

フォルダ入力では、`--input` と `--output` の両方がフォルダである必要があります。Jasna は画像を先に、その後で動画を処理し、全体の `[current/total]` ファイルカウンターを表示します。デフォルトでは出力フォルダに `<name>_out<ext>` を書き込みます。

```bash
jasna --input input_folder --output output_folder
```

フォルダ処理では、GUI と同じ `{original}` ファイル名テンプレートも使えます:

```bash
jasna --input input_folder --output output_folder --output-pattern "{original}_restored.mp4"
```

画像は元の拡張子を維持し、動画はテンプレートに拡張子がある場合その拡張子を使います。Jasna は処理前に予定される出力パスを確認し、テンプレートによって複数の入力が同じ出力ファイルに対応する場合はエラーで終了します。

## エクスポート後のアクション

GUI では、キュー全体が完了した後に実行するアクションを選べます: **なし**、**PC をシャットダウン**、**カスタムコマンド**。同じ機能は Windows と Linux の CLI でも使えます:

```bash
jasna --input input.mp4 --output output.mkv --post-export-action shutdown
```

カスタムコマンドは、すべてのエクスポート完了後にシステムシェルで実行されます:

```bash
jasna --input input_folder --output output_folder --post-export-action command --post-export-command "echo done"
```

## 初回実行

初回実行は、お使いの GPU 向けに TensorRT エンジンをコンパイルするため遅くなります。通常 **15-60 分**かかります。

コンパイル中はブラウザを含む他のアプリを閉じ、PC の使用を避けてください。エンジンは `model_weights` にキャッシュされ、以降の実行で再利用されます。古い Jasna バージョンから新しいバージョンへ、エンジンファイルやフォルダをコピーできます。

処理中に VRAM 不足になる場合は、まず **max clip size** を下げてください。例: `180` から `60`。BasicVSR++ コンパイルを無効化してもピーク VRAM は下がりますが、処理は遅くなります。

## モデルの選び方

### 検出モデル

通常は最新の RF-DETR モデルを使ってください。Lada YOLO モデルも利用でき、2D アニメーションではより良い場合があります。

CLI オプション:

```bash
jasna --input input.mp4 --output output.mkv --detection-model rfdetr-v5
```

### セカンダリ復元

Jasna と Lada は各モザイク領域の 256x256 クロップを復元します。そのため、大きなモザイク領域、クローズアップ、4K 動画では、一次復元後にぼやけて見えることがあります。セカンダリ復元モデルを使うと、復元済みクロップを 512x512 または 1024x1024 にアップスケールしてから元の映像へ合成できます。

対応しているセカンダリモデル:

- **unet-4x**: 支援者モデル。現在のテストでは TVAI より高速で同程度の品質です。JAV ドメイン内データセットで訓練されており、見た目は TVAI `iris-2` に近いです。[unet-4x / セカンダリ復元の例（SLS Discord）](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260) を確認できます。支援者キーで解除します。詳しくは[プロジェクトを支援する](#プロジェクトを支援する)をご覧ください。品質問題がある場合は [GitHub issue](https://github.com/Kruk2/jasna/issues) を開いてください。
- **RTX Super Resolution**: 非常に高速で無料、追加依存関係はありません。品質はそれなりです。一部の動画ではフリッカーが出る場合があるため、短いクリップで先に試してください。
- **TVAI**: 現在のテストでは RTX Super Resolution より高品質で unet-4x と同程度ですが、非常に遅いです。[Topaz Video](https://www.topazlabs.com/topaz-video) が必要です。有料で Windows のみです。推奨モデルは `iris-2` です。
- **FlashVSR**（`+modi`、オフライン、実験的）: 強いテクスチャを復元する拡散 4x アップスケーラ。12〜16GB VRAM が一次パイプラインと衝突しないよう、オフライン 3 段パスで実行します。利用者が用意する `FlashVSR_plus` checkout と uv-managed venv が必要。ファイル出力専用。参照: [docs/FLASHVSR_ja.md](docs/FLASHVSR_ja.md)。

CLI オプション:

```bash
jasna --input input.mp4 --output output.mkv --secondary-restoration unet-4x
```

TVAI では、`--tvai-args` で Topaz モデルのパラメータをカスタマイズできます。デフォルトモデルは `iris-2` です。Topaz Video 用に次の環境変数を設定してください:

<img width="505" height="37" alt="Topaz Video environment variables" src="https://github.com/user-attachments/assets/e19ced9d-d549-4e85-b20f-888e42466f1d" />

VRAM と処理時間:

| セカンダリ種別 | CAWD 1080p | KV-109 1080p |
| --- | ---: | ---: |
| セカンダリなし | 22秒 / 10.0 GB VRAM | 11秒 / 10.7 GB VRAM |
| unet-4x | 29秒 / 12.5 GB VRAM | 14秒 / 12.6 GB VRAM |
| RTX Super-Res | 25秒 / 11.7 GB VRAM | 13秒 / 11.4 GB VRAM |
| TVAI (2 workers, Iris-2) | 52秒 / 12.1 GB VRAM | 24秒 / 12.4 GB VRAM |

復元例は [SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1516497879684874260) にあります。

### 静止画復元

静止画では、動画パイプラインの代わりに、ファインチューニング済みの Stable Diffusion 1.5 inpaint モデルを使えます。モザイクを検出し、各領域を 512x512 で inpaint して、結果を元画像へ合成します。

- CLI: `jasna --input photo.png --output out.png`
- GUI: 画像をキューに追加します。画像ジョブは自動的に SD 1.5 にルーティングされます。
- 調整オプション: `--sd15-steps`、`--sd15-strength` (`<= 0.7` に制限)、`--sd15-freeu` / `--no-sd15-freeu`、`--sd15-seed`、`--sd15-variants N`。
- 画像モデルは `--image-restoration-model-name` で選択します。現在のデフォルトかつ唯一の値は `sd-15-jav` です。
- `--restoration-model-name` は動画専用です。

SD 1.5 モデルは**同梱されておらず**、約 **6.9 GB** です。配置先は `model_weights/sd-15-jav/` です。自分でそこにバンドルを置くか、Jasna に [huggingface.co/Kruk2/sd-15-jav](https://huggingface.co/Kruk2/sd-15-jav) から取得させることができます。ダウンロード前に、CLI プロンプトまたは GUI の **Download model** ボタンで確認されます。

checkpoint は現在、支援者のみ利用でき、unet-4x と同じキーを使います。詳しくは[プロジェクトを支援する](#プロジェクトを支援する)をご覧ください。

SD 1.5 経路は実験的です。結果はシーンによって変わりますが、うまく合う画像では非常に良い結果になることがあります。複数の `--sd15-variants` を試し、最も良い結果を残してください。推論中は約 **7 GB VRAM**、大きな 4K 画像ではもう少し多く必要です。

例は [SLS Discord](https://discord.com/channels/1196376491815092265/1199059436199759943/1492139124348420106) と[追加の SD 1.5 例](https://discord.com/channels/1196376491815092265/1199059436199759943/1516571355317800990)にあります。

## 品質と VRAM の調整

### Max Clip Size と Temporal Overlap

テンポラルオーバーラップはクリップ境界のフリッカーを軽減します。値を大きくすると処理時間は増えますが、フリッカーを減らせる場合があります。`20` を超えても通常は大きな効果はありません。

推奨開始点:

- GPU が扱える範囲で最も大きい **max clip size** を使います。
- **temporal overlap** を `8` から `20` の間に設定します。
- `--enable-crossfade` でクロスフェードを有効のままにします。

限られたテストでの目安:

| Max clip size | Temporal overlap | メモ |
| ---: | ---: | --- |
| 60 | 6 | VRAM を抑えたい場合。 |
| 90 | 8 | 現在のデフォルト寄りのバランス。 |
| 180 | 15 | BasicVSR++ コンパイル有効時は 12 GB+ VRAM が必要。無効時は少なくなります。 |

4K 動画はより多くの VRAM を使います。低いクリップサイズでも同程度の品質になり、処理が速くなる場合があります。`60` 未満のクリップサイズでも動画によっては動きますが、モデルコンパイルを無効にしてでも `60` を推奨します。

CLI 例:

```bash
jasna --input input.mp4 --output output.mkv --max-clip-size 90 --temporal-overlap 8 --enable-crossfade
```

### 復元モデルのコンパイル

復元モデルは TensorRT サブエンジンにコンパイルされます。コンパイルすると高速になりますが、より多くの VRAM を使います。パフォーマンスと引き換えに無効化できます:

```bash
jasna --input input.mp4 --output output.mkv --no-compile-basicvsrpp
```

以下はコンパイル済みエンジンのみの VRAM であり、総処理 VRAM ではありません:

| | Clip 60 | Clip 180 |
| --- | ---: | ---: |
| Engine VRAM, compiled | 約 1.9 GB | 約 5.4 GB |
| Engine VRAM, no compilation | 約 1.2 GB | 約 1.2 GB |

## ストリーミング

ストリーミングでは、ファイル全体を先に処理せず、復元動画をリアルタイムで視聴できます。

### ブラウザプレーヤー

ストリーミングモードは現在 CLI のみです。ブラウザで HLS プレーヤーが開きます。動画ファイルを選んで視聴を開始できます。シークに対応しています。

```bash
jasna --stream
```

Windows では、ストリーミングもアプリ本体と同じファイルを使います: `jasna.exe --stream`。別の `jasna-cli.exe` がない場合があります。

### Stash 連携

Jasna はカスタム Stash フォーク経由で [Stash](https://github.com/stashapp/stash) 内から使用できます。シーンを再生すると Stash が Jasna を自動起動し、視聴しながら処理します。シークも動作します。

カスタムフォーク: **[Stash v0.30.1-jasna](https://github.com/Kruk2/stash/releases/tag/v0.30.1-jasna)**

セットアップ:

1. 上記リンクから Stash フォークをダウンロードします。
2. Stash 起動前に環境変数を設定します:
   - `JASNA_CLI_PATH`: `jasna.exe` のフルパス。自分でリネームした場合はその名前にします。
   - `JASNA_WORKING_DIR`: その実行ファイルがあるフォルダのフルパス。
3. **重要:** Stash を使う前に、同じ設定で短い動画を一度ストリーミングしてください。TensorRT エンジンを事前コンパイルでき、初回のヘルスチェックタイムアウトを避けやすくなります。
4. Stash を起動してシーンを再生します。

Stash ログに `timeout waiting for jasna-cli to become healthy` と出る場合は、まず `JASNA_CLI_PATH` を確認し、その後で上の方法で事前コンパイルしてください。

## ベンチマーク

RTX 5090 + i9 13900k:

| ファイル | クリップ (秒) | lada 0.10.1 | jasna 0.3.0 | jasna 0.5.0 | **jasna 0.6.2** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ABF-017** (4k, 2時間25分) | 60 | 02:56:26 | 01:20:49 (2.2倍高速) | 01:10:00 (2.5倍高速) | xx |
| **HUBLK-063** (1080p, 3時間10分) | 180 | 01:34:51 | 44:21 (2.1倍高速) | 37:57 (2.5倍高速) | **30:58 (3.1倍高速)** |
| **DASS-570_2m** | 30 | 01:08 | 00:30 (2.3倍高速) | 00:24 (2.8倍高速) | **00:20 (3.4倍高速)** |
| **NASK-223_Test** | 30 | 03:12 | 01:18 (2.5倍高速) | 01:02 (3.1倍高速) | **00:58 (3.3倍高速)** |
| **test-007** | 30 | 01:16 | 00:41 (1.9倍高速) | 00:28 (2.7倍高速) | **00:22 (3.5倍高速)** |
| **厚码测试2** | 30 | 01:52 | 00:43 (2.6倍高速) | 00:36 (3.1倍高速) | **00:34 (3.3倍高速)** |

## プロジェクトを支援する

支援は追加モデルの訓練に使われます。主に、大きな GPU のレンタル代と、大きなデータセットで訓練するための計算時間です。支援者には以下を解除するキーが提供されます:

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

## 現在の制限と TODO

Jasna は開発初期段階です。主な目標は、復元品質、モザイク検出、速度、VRAM 使用量をこの順で改善することです。現在はより技術的なユーザー向けのため、一部のワークフローはまだ粗い場合があります。プルリクエストは歓迎します。

現在の TODO:

- 適切な VR サポート。
- SeedVR サポート。
- パフォーマンスと VRAM 使用量の継続的な改善。

## ソースから実行

> **`+modi` ビルドガイド:** このフォークはネイティブ GPU ライブラリをビルドし、Jasna を**ソースから実行**します。公開のパッケージ済み/凍結バイナリはありません — パッケージングツールはプライベートな `jasna/protection` サブモジュールにあります（上流と同じ構成）。CUDA 13.0 ツールチェーン、ネイティブライブラリ、ffmpeg 8 / mkvmerge、TensorRT エンジン設定を網羅した手順:
> - Linux: [docs/BUILDING_LINUX_ja.md](docs/BUILDING_LINUX_ja.md)（[English](docs/BUILDING_LINUX_en.md)）
> - Windows: [docs/BUILDING_WINDOWS_ja.md](docs/BUILDING_WINDOWS_ja.md)（[English](docs/BUILDING_WINDOWS_en.md)）

`pyproject.toml` の Python 要件: **Python 3.13 以上**。

公開されているソースチェックアウトには protection module が含まれていません。開発や無料モデルには使えますが、通常のソースチェックアウトでは **unet-4x** や **SD 1.5 画像復元** などの支援者限定モデルは利用できません。

実行時依存関係をインストールします:

```bash
uv pip install . --no-build-isolation
```

Nvidia ライブラリをビルドするには、以下も必要です:

- C++ 対応の VS Build Tools 2022。
- システムにインストールされた CUDA 13.0。
- `cmake` と `ninja`:

```bash
uv pip install cmake ninja
```

開発者セットアップには以下も必要です:

- `ffmpeg` と `ffprobe` を `PATH` に配置。`ffmpeg` のメジャーバージョンは **8** である必要があります。
- [MKVToolNix](https://mkvtoolnix.download/downloads.html) に含まれる `mkvmerge`。
- 次の 2 つのライブラリを Python 環境にインストール:
  - https://codeberg.org/Kruk2/vali
  - https://codeberg.org/Kruk2/PyNvVideoCodec

その後、Jasna を editable mode でインストールします:

```bash
uv pip install -e .[dev]
```
