# 変更点サマリ: `modi`（`0.11.2+modi`）vs upstream

このブランチが upstream [Kruk2/jasna](https://github.com/Kruk2/jasna) に対して加えている変更を、「いつ追加したか」ではなく「**今、何が違うか**」で整理したものです。

- ブランチ: `modi`（fork `https://github.com/sh202603/jasna.git`）
- バージョン: `0.11.2+modi`。upstream `b623466f`（`v0.11.2` タグ）にリベース済み
- リベース方針: **重複する機能は upstream 優先。** upstream が自前で実装した項目はブランチから外し、本文ではなく[付録 A](#付録-a-upstream-に吸収されたまたは-drop-した変更)に記載します。

本書の読み方:

| 節 | 内容 |
| --- | --- |
| [1. この公開ビルドの範囲](#1-この公開ビルドの範囲) | 公開ソースではできないこと（支援者モデル、LTX） |
| [2. 機能一覧](#2-機能一覧) | modi の全機能・切り替え方・状態・詳細ドキュメントを 1 表に |
| [3. 復元モデル](#3-復元モデル) | SeedVR2 一次復元、FlashVSR / SwiftVR / SwiftVR distill 二次復元、RTX Super Res の追加オプション、LTX 出力の鮮明化 |
| [4. フレーム生成](#4-フレーム生成) | `--frame-gen`、`rife` / `rtx` バックエンド、`jasna-framegen` |
| [5. メディアとエンジンのバックエンド](#5-メディアとエンジンのバックエンド) | torchcodec、cuDNN FP8 upsample、TensorRT-RTX フレーバー |
| [6. GUI](#6-gui) | A/B モデル比較、動画プレーヤーのチェックポイント選択、modi 機能の設定項目 |
| [7. このブランチで保持しているバグ修正](#7-このブランチで保持しているバグ修正) | upstream に無い修正 |
| [8. ビルド・依存・パッケージング](#8-ビルド依存パッケージング) | pyproject の差分、重みの解決、パッチ、Nuitka バンドル、ビルドガイド、既知の未対応 |
| [9. 機能の組み合わせ](#9-機能の組み合わせ) | 互換マトリクス |
| [付録 A](#付録-a-upstream-に吸収されたまたは-drop-した変更) | upstream に吸収された、または drop した旧 modi 変更 |
| [付録 B](#付録-b-リベース履歴) | 各リベースで取り込んだ upstream の変更と fork 側の適合（新しい順） |

各機能には設計・計測・検証記録を載せた専用ドキュメントがあります。本書は「何か・どう有効にするか・何を拒否するか・現状」を述べ、数値は要点だけに絞っています。

---

## 1. この公開ビルドの範囲

- **支援者モデルは不活性（inert）。** `unet-4x`、SD 1.5 画像復元、有効化フローは非公開の `jasna/protection` サブモジュールを要し、この公開フォークには含まれません。upstream のコードは同梱されますが動作しません。これらのモデルは upstream の公式リリースを使ってください。
- **LTX による復元はできません。** LTX の重みは暗号化（`.enc`）されて配布され、復号できるのは公式リリースだけです。`--restoration-model-name ltx` / `ltx-undistilled` と GUI の LTX モードは動かず、`--ltx-trial` はランダム重みで速度を測るだけです。LTX の公開部分（モザイクのスキャンと合成）を、公式リリースが復元した動画の鮮明化 [`--ltx-crop-vsr`](#36-ltx-出力の鮮明化--ltx-crop-vsr) が再利用しています。
- **それ以外は動作します**: 検出、BasicVSR++ 復元、RTX Super Res と TVAI の二次復元、区間エディター、VR180、ストリーミング、エクスポート後アクション、そして以下の modi 機能。
- **ソースから実行します。** セットアップはビルドガイド（[Linux](building_linux.md)、[Windows](building_windows.md)）に従います。Windows 向けには `scripts/build_nuitka.py` で実験的なバンドルを作れます（[frozen_build.md](frozen_build.md)）。

## 2. 機能一覧

「検証済み」は、その OS の実機で e2e（入力から出力まで）走らせたことを指します（Linux: RTX 5080 16 GB、Windows 11: RTX 5060 Ti 16 GB または RTX 5080 16 GB）。

| 領域 | 機能 | 切り替え | 状態 | 詳細 |
| --- | --- | --- | --- | --- |
| 一次復元 | SeedVR2 + LoRA | `--restoration-model-name seedvr2` | 実験的。Linux・Windows 検証済み、`--seedvr2-accel` も両 OS | [seedvr2.md](seedvr2.md)、[3.1](#31-seedvr2--lora-一次復元) |
| 二次復元 | FlashVSR（オフライン 3 段 / inline） | `--secondary-restoration flashvsr` / `flashvsr-inline` | 実験的。Linux・Windows 検証済み | [flashvsr.md](flashvsr.md)、[3.2](#32-flashvsr-二次復元) |
| 二次復元 | SwiftVR（inline / オフライン 3 段） | `--secondary-restoration swiftvr-inline` / `swiftvr` | 実験的。Linux・Windows 検証済み | [swiftvr.md](swiftvr.md)、[3.3](#33-swiftvr-二次復元) |
| 二次復元 | SwiftVR distill（プロセス内の小型 student） | `--secondary-restoration swiftvr-distill` | 実験的。Linux・Windows 検証済み | [swiftvr.md](swiftvr.md#swiftvr-distill--secondary-restoration-swiftvr-distill)、[3.4](#34-swiftvr-distill-二次復元) |
| 二次復元 | RTX Super Res の追加オプション | `--rtx-strength`、`--rtx-highbitrate` | オプトイン。目視確認済み | [3.5](#35-rtx-super-res-の追加オプション) |
| LTX 後処理 | LTX 出力を SwiftVR で鮮明化 | `--ltx-crop-vsr swiftvr` / `swiftvr-distill` | Linux・Windows 検証済み、ユーザー目視合格 | [ltx_crop_vsr.md](ltx_crop_vsr.md)、[3.6](#36-ltx-出力の鮮明化--ltx-crop-vsr) |
| フレーム生成 | 2x / 4x のフレームレート倍化 | `--frame-gen 2x\|4x`、`--frame-gen-backend rife\|rtx`、`jasna-framegen` | Linux・Windows 検証済み | [frame_generation.md](frame_generation.md)、[4](#4-フレーム生成) |
| メディアバックエンド | torchcodec のデコード/エンコード | `--video-backend`、`--decode-backend`、`--encode-backend` | 実験的。Linux・Windows 検証済み | [torchcodec_backend.md](torchcodec_backend.md)、[5.1](#51-torchcodec-動画バックエンド) |
| エンジンバックエンド | cuDNN FP8 の upsample 段 | `--fp8-recon` | 実験的。Linux・Windows 検証済み | [fp8_recon.md](fp8_recon.md)、[5.2](#52-cudnn-fp8-upsample-バックエンド--fp8-recon) |
| エンジンフレーバー | TensorRT-RTX（JIT エンジン） | `nvidia-rtx` extra を導入 | Linux・Windows 検証済み | [tensorrt_rtx.md](tensorrt_rtx.md)、[5.3](#53-tensorrt-rtx-フレーバーnvidia-rtx-extra) |
| GUI | A/B モデル比較 | 区間エディターの **A/B 比較…** | Linux・Windows 検証済み | [segments.md](segments.md#ab-モデル比較)、[6.1](#61-ab-モデル比較区間エディター) |
| GUI | 動画プレーヤーの復元チェックポイント選択 | プレーヤーの設定カード | Linux・Windows 検証済み | [gui.md](gui.md#動画プレーヤー)、[6.2](#62-動画プレーヤーの復元チェックポイント選択) |
| パッケージング | Windows 向け Nuitka バンドル | `scripts/build_nuitka.py` | 実験的、Windows のみ | [frozen_build.md](frozen_build.md)、[8](#8-ビルド依存パッケージング) |
| ビルド | ソースからのセットアップガイド | — | 本ベースで Linux と Windows を検証済み（2026-10-18） | [building_linux.md](building_linux.md)、[building_windows.md](building_windows.md) |

外部モデル系の機能は GUI には出ず、CLI 専用です。GUI に出るのはフレーム生成、動画バックエンド、FP8 トグル、RTX Super Res の追加オプション、A/B 比較、プレーヤーのチェックポイントです（[6.3](#63-modi-機能の設定項目)）。

---

## 3. 復元モデル

### 3.1 SeedVR2 + LoRA 一次復元

`--restoration-model-name seedvr2` は、一次復元器 BasicVSR++ を SeedVR2 3B（one-step SR diffusion）+ 自作 LoRA（モザイクをセル平均化劣化として除去することを学習。rank 16、約 90 MB、[sh202603/lada-seedvr2-lora](https://huggingface.co/sh202603/lada-seedvr2-lora)、AGPL-3.0）に差し替えます。検出された 256px クロップを scale 1 のまま diffusion forward に通し、検出・クロップ抽出・blend・encode は不変で、既定の `basicvsrpp` 経路は bit 単位で変わりません。位置づけ: 低解像度で劣化の大きい素材に最も強く（BasicVSR++ が均す入力からも diffusion の事前分布がディテールを再構成する）、原信号の復元ではなく「ありそうなディテール」の付与なので素材に応じて使い分け、既定は `basicvsrpp` のままです。本フォークの lada-ex 実装からの移植。

- **プロセスモデル。** 利用者が用意する [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) venv 内の常駐 worker（`jasna/restorer/seedvr2_lora_worker.py`、jasna import なし）と、stdin/stdout 上の JSON ヘッダ + 生 uint8 BGR で 1 クリップずつ往復します。死亡時は respawn + 同一クリップ 1 回リトライ、素通しフォールバックなし（スキップはモザイク残存を意味するため）。内部は peft による LoRA 注入、33 フレーム / stride 24 のスライディングウィンドウを GPU 常駐の float32 アキュムレータへクロスフェード合成して最後に 1 回量子化、入力モザイククロップ参照のクリップ単位 LAB 色転写、総 VRAM で自動切替するクリップ毎 `empty_cache`（`--seedvr2-empty-cache`、`auto` は総 VRAM 20 GiB 未満のみ解放）。
- **jasna 側の適応。** この復元器ではクロップのパディングを reflect でなくゼロ埋めにする（`pad_mode`。LoRA の訓練分布）、ワイヤは BGR のまま親アダプタで RGB 変換、そして BasicVSR++ を bit 不変に保つため復元器の属性でゲートした blend 側の 2 つの revert: `_revert_valid_edges`（VAE/DiT がゼロパッドの黒を有効領域 2〜3px に滲ませ、bbox 縁の暗線として合成されていた）と `_revert_outside_mask`（1-step diffusion はクロップ全体を再合成するため blend のフォールオフ環帯に低周波のずれが乗っていた。検出マスクの信頼ゾーン外をすべて入力へ戻し、blend mask 自身のフォールオフを重みに使う。クロップを埋める大型モザイクにはクロップ境界ランプを追加）。
- **高速化**（`--seedvr2-accel`、既定 off）: VAE の融合経路（`fused_vae`、comfy-kitchen のカーネル）と FP8 DiT（`fp8_dit`。LoRA を先に基底重みへマージ）。どちらも fork [sh202603/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/sh202603/ComfyUI-SeedVR2_VideoUpscaler) の `modi` ブランチが提供します。worker は使えない部品を外して理由を ready ハンドシェイクで返します（upstream の checkout では両方外れ、出力は off と bit 一致）。2 分素材の実行全体の壁時計: Linux RTX 5080 で 1.68〜1.77 倍速（480p 117 → 70 秒、1080p 220 → 126 秒、4K 272 → 154 秒）、Windows RTX 5080 で 1.45〜1.57 倍速。GPU 全体の VRAM ピークは 3.1〜3.7 GiB 減。出力は off に対して PSNR 46〜50 dB で目視同等。
- **フラグ。** `--seedvr2-repo`（必須）、`--seedvr2-python`、`--seedvr2-lora`、`--seedvr2-window` / `--seedvr2-overlap`（33 / 9）、`--seedvr2-color-fix {none,lab,wavelet}`、`--seedvr2-empty-cache {auto,always,never}`、`--seedvr2-accel`、base 重みの上書き `--seedvr2-model-dir`、`--seedvr2-dit`、`--seedvr2-lora-rank`。
- **制約。** VR モード、`--frame-gen`、`flashvsr-inline`、`swiftvr-inline` は起動時に拒否（常駐 worker 2 つは 16 GB を超える）。TVAI と `--stream` は警告。オフラインの `flashvsr` / `swiftvr` は併用可（Phase が同時常駐しない）で、これが最高品質構成です。seedvr2 選択時は BasicVSR++ のエンジンコンパイルをスキップします。v0.11.1 ベースからは `seedvr2` が upstream の復元モデル選択（`CLI_RESTORATION_MODELS`、`provide_restoration_models`）の 1 項目になりました。
- **状態。** ユニット `tests/test_seedvr2_lora.py` + `test_main.py` の検証マトリクス。Linux・Windows で e2e 検証済み（出力フレーム数 = 入力、worker 正常終了。480p 全編モザイクで全体 VRAM ピーク 12.6 GB）。A/B 比較（[6.1](#61-ab-モデル比較区間エディター)）でも seedvr2 を側として選べます。

### 3.2 FlashVSR 二次復元

256px の一次復元クロップを one-step diffusion VSR の [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) で拡大して再 blend し、BasicVSR++ が大きなモザイク・接写・4K でぼやけさせるテクスチャを補う二次復元です。処理は 1024px（4x、モデルネイティブ）または 512px（`--flashvsr-scale 2`）で、出力解像度は変わりません。checkout・重み・専用 venv は利用者が用意して `--flashvsr-repo` で渡します。fork [`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus) を推奨（tiny-long の修正と高速化を含み、`uv sync` で依存が揃う）。venv の基底 Python には開発ヘッダが必要です（Triton の実行時 JIT）。モードは 2 つ:

- **`flashvsr`（オフライン 3 段）。** `dump → FlashVSR → reblend` を独立サブプロセスで順に実行し、各段の終了で VRAM を全解放するため、tiny モードの 12〜16 GB のピークが一次の約 9 GB と重なりません。中間 bundle をディスクに持ち（モザイク多めの長尺で数百 GB 級、scale 2 で 1/4）、`--flashvsr-bundle-dir` で段階再開できます。12 GB 級 GPU で動作。encode は 2 回。`jasna/restorer/flashvsr_offline.py` + `flashvsr_phase2_driver.py`、サブプロセス分岐は `jasna/__main__.py` の `--flashvsr-phase`。
- **`flashvsr-inline`（単一パス）。** FlashVSR の **tiny-long**（clip 長に依存せず定常 ~11.9 GB）を、[`--fp8-recon`](#52-cudnn-fp8-upsample-バックエンド--fp8-recon)（自動有効化）で ~1.6 GB に絞った一次と同時常駐させます。同期 `SecondaryRestorer` が FlashVSR venv の worker を常駐起動し、length-prefixed の uint8 BGR プロトコルでクロップを往復します（worker は lada-ex の `flashvsr_worker.py` と逐語同一に保ち、RGB 反転は jasna 側アダプタ）。`--flashvsr-tiles {1..4}` は DiT を横短冊ごとに回す VRAM 調整。16 GB カードが必要。`jasna/restorer/flashvsr_inline_secondary_restorer.py` + `flashvsr_inline_worker.py`。

共通の挙動（特記なければ両モード）:

- **色補正は常時適用**: 各出力クロップを一次出力の bicubic 拡大に wavelet 再構成（高周波 = FlashVSR、低周波 = 参照）で合わせ、合成領域と周囲の色調差を除きます（変わった画素の median |Δmean| 1.7 → 0.25）。`JASNA_FLASHVSR_COLOR_FIX=adain|wavelet|none` は検証専用の上書き。
- **clip 長**: inline は `--max-clip-size`（既定 90）をそのまま使い、オフラインの `--flashvsr-max-clip-frames` 既定も 90。かつての 32 フレーム上限は overlap の再処理で FlashVSR 時間を約 2 倍にしていました。
- **worker が全 clip を検査**: tiny-long は chunk ループの例外を握り潰すため、VRAM 天井での clip 途中 OOM が末尾フレームの複製で埋められた clip として返っていました（残像）。worker は戻り値と枚数を検査し、キャッシュ解放後に 1 回リトライ、再失敗なら clip エラーで停止します。結果として 16 GB では scale 4 の tiles なしは使えず、`--flashvsr-tiles 2` を使います。
- **`--flashvsr-accel`**（off）: fork の FP8 projector 畳み込み・FP8 DiT Linear・融合カーネル。約 1.4 倍速でピーク 1.4 GiB 減。RTX 40 系未満・fp16・`-v 10`・upstream の checkout では部品ごとに自動で標準処理へ戻り、全部品が外れた出力は off と bit 一致。有効な部品は ready ハンドシェイク、実行中の降格は clip 応答で jasna に返ります。
- **`--flashvsr-lora`**（inline 専用）: 公開している `lada_flashvsr_secondary_lora_v1.pt` を DiT に bf16 アダプタとして適用し、FlashVSR の中帯域の過鮮鋭（4〜8 px 帯で約 −30%）と過剰な粒を抑えます。`--flashvsr-accel` と併用可。オフラインは拒否（主用途の SeedVR2 一次では FlashVSR の足す粒の大半を消すため）。
- **tiny-long パッチ**: inline には multi-chunk の修正（per-chunk のキャッシュ clear 2 箇所の無効化）が必要です。fork は含み、upstream の checkout には [`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../../patches/flashvsr_plus_tinylong_multichunk_fix.patch) を当てます。restorer は起動時に検査し、無ければ明示エラーで停止します。
- **フラグ**: `--flashvsr-repo`（必須）、`--flashvsr-python`、`--flashvsr-model-dir`、`--flashvsr-version` / `--flashvsr-dtype`、`--flashvsr-scale {2,4}`、`--flashvsr-tiles`、`--flashvsr-accel`、`--flashvsr-lora`、オフライン用の `--flashvsr-max-clip-frames`、`--flashvsr-unload-dit`、`--flashvsr-tiled-vae`、`--flashvsr-bundle-dir`、`--flashvsr-keep-bundle`。
- **制約**: ファイル出力専用（`--stream` と画像入力は不可）、`--frame-gen` 不可（別パスで実行）。オフラインはさらに `--retarget-high-fps`、`--segments`、VR を拒否。inline は SeedVR2 一次と排他、オフラインは併用可。inline はモザイクが多い区間で FlashVSR の約 15 crop-fps に律速されます。
- **状態**: ユニット `tests/test_flashvsr_{offline,inline}.py` + `test_main.py`。Linux・Windows で両モード・両 scale・480p と 1080p の e2e 検証済み（フレーム数 = 入力、変化はモザイク領域のみ）。要点の数値（Linux、480p、4930 フレーム、clip 90）: inline scale 2 / tiles 1 は 113 秒・GPU 全体ピーク 10.1 GB、scale 4 / tiles 2 は 491 秒・13.7 GB、オフライン scale 2 は 194 秒・7.8 GB。Windows 16 GB の 1080p は tiles 2 を推奨。計測と色補正ゲートは [flashvsr.md](flashvsr.md)。

### 3.3 SwiftVR 二次復元

`--secondary-restoration swiftvr-inline`（inline）と `swiftvr`（オフライン 3 段）は、FlashVSR と同じ役割を [SwiftVR](https://github.com/H-oliday/SwiftVR)（one-step streaming diffusion VSR、Wan2.2-TI2V-5B バックボーン、Apache-2.0）で行います。fork [`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR) の `modi` ブランチが必須です（フレーム数を保つメモリ上の clip API `SwiftVRPipeline.restore_clip()` を追加）。クロップは 1024px（4x）または 512px（`--swiftvr-scale 2`）へ。checkout、約 20 GB のチェックポイント、`uv sync` の venv は利用者が用意します（`--swiftvr-repo`）。同じマシンで SwiftVR inline は FlashVSR inline より 1080p で約 2.4 倍（scale 2）〜3.3 倍（scale 4）速いです。

- **inline** は `flashvsr-inline` と同じ同期 `SecondaryRestorer` と BGR ワイヤで、SwiftVR venv の worker（`jasna/restorer/swiftvr_inline_worker.py`）を使います。SwiftVR は固定チャンク（先頭 28、以降 24 フレーム）で処理するので VRAM は clip 長に依存せず、clip 上限はありません。短い clip は `restore_clip()` の中で埋めます。
- **オフライン 3 段（`swiftvr`）** は FlashVSR の orchestrator（`flashvsr_offline.py`）を engine レコード（`OfflineEngine`）で共用します。Phase 1、Phase 3、bundle（version 2、`view_placements` を持つ）、ディスク検査、`--flashvsr-phase` フックは同じで、Phase 2 は SwiftVR venv の `swiftvr_phase2_driver.py` が inline worker の関数を再利用します。ここでは bf16 が正規の経路（SwiftVR 段が GPU を独占）なので、12 GB 級 GPU、FP8 非対応 GPU、SeedVR2 一次向けのモードです。Phase 1 に `--fp8-recon` を渡せば、ロスレス encode は inline と全フレーム bit 一致します。
- **高速化は既定 on（`--swiftvr-accel`、`--no-swiftvr-accel` で無効）**: fork の FP8 DiT と torch.compile。worker はモデルを読む前に GPU 世代（sm89 以上）と Triton のカーネルビルドを確かめ、使えない部品を外して理由をハンドシェイクで返します。FP8 が外れると DiT は bf16（約 12 GiB）で動き 16 GB では一次と同居できないため、jasna は警告して続行し、VRAM 不足なら OOM で失敗します。
- **色補正は常時適用**（wavelet、一次出力の bicubic 拡大を参照）、GPU 上でフレームごとに適用。primitive は FlashVSR worker を path 読み込みで共有し、そのファイルは lada-ex と逐語同一のままにします。`JASNA_SWIFTVR_COLOR_FIX` は検証専用の上書き。
- **切り出し view の平滑化（`--swiftvr-view-window`、既定 15、`0` で無効）**: scale 2 の揺れの主因は、SwiftVR に渡す切り出しの枠がフレームごとに動くこと（クロップは検出枠に追従し、SwiftVR はサブピクセルのずれに反応してディテールを描き直す）でした。一次段の末尾で各フレームの 256 格子を前後 N フレームの移動平均の配置（view）へ再サンプルして渡し、blend は view から元フレームへ直接リサンプルします。crop 単位の flow-warping error 比は scale 2 で 4.99 → 2.57（scale 4 は 2.16）。一次復元、mask、worker、ワイヤは不変で、無効時と他の二次復元は bit 不変。`jasna/tracking/crop_view.py`、`restoration_pipeline.py`、`blend_buffer.py`、`pipeline_items.py`。原因の特定と対策は [mioh-labs](https://github.com/mioh-labs/mioh) の報告によります。
- **フラグ**: `--swiftvr-repo`（必須）、`--swiftvr-python`、`--swiftvr-model-dir`、`--swiftvr-scale {2,4}`、`--swiftvr-view-window`、`--swiftvr-accel`、オフライン専用の `--swiftvr-bundle-dir` / `--swiftvr-keep-bundle`。dtype、version、tiles、LoRA、max-clip-frames はありません。
- **制約**: FlashVSR の各モードと同じ起動時検査（inline: fp8-recon 自動有効化、`--frame-gen` 拒否、SeedVR2 一次と排他。オフライン: `--stream`、VR、`--segments`、`--frame-gen`、`--retarget-high-fps`、フォルダ/画像入力を拒否、SeedVR2 一次は可）。GUI には出ません。
- **状態**: `tests/test_swiftvr_inline.py`、`tests/test_swiftvr_offline.py`、`test_main.py`。Linux・Windows で両モード・両 scale・480p と 1080p の e2e 検証済み。目視 A/B（利用者）では scale 4 の質感は FlashVSR と同等、view 平滑化後の scale 2 は時間安定性で scale 4 と同等。GPU 全体ピーク（Linux inline）: scale 4 FP8 で 12.6 GB、scale 2 で 10.3 GB。全計測、Windows の注意事項、「時間安定性を決める要因」の調査は [swiftvr.md](swiftvr.md)。

### 3.4 SwiftVR distill 二次復元

`--secondary-restoration swiftvr-distill` は、SwiftVR の出力から蒸留した小さな畳み込みの student（`TinyROIEnhancer`、24 チャンネル、残差ブロック 12 個、約 13 万パラメータ、チェックポイント `roi-distill-pilot-v1`）を SwiftVR の代わりに動かします。256px の一次クロップを 5 フレームずつ入れ、中央フレームを 512px（2x）で受け取り、`--swiftvr-scale 2` と同じく再 blend します。jasna のプロセス内で PyTorch FP32 で動き、checkout、venv、worker は不要です。重みは [mioh](https://github.com/mioh-labs/mioh) の作者が学習して [`okatti/swiftvr-distill`](https://huggingface.co/okatti/swiftvr-distill) に公開したもの（AGPL-3.0）。同梱せず、`model_weights/` に `swiftvr-distill.pt` を置きます（別の場所なら `--swiftvr-distill-model`）。

- **復元器**: `jasna/restorer/swiftvr_distill_secondary_restorer.py`（中央フレーム 4 枚ずつのバッチ、5 フレーム窓は clip の端を繰り返す）。`swiftvr_distill_model.py` のモデル定義は公開 README どおり（3x3 conv 15→24 と ReLU、`x + 0.1 * conv(relu(conv(x)))` のブロック 12 個、3x3 conv 24→12 と PixelShuffle(2)、出力 = 中央フレームの双線形 2 倍 + その細部）。ローダーは検査済みの dict だけを `weights_only=True` で受け付けます。
- **フラグ**: `--swiftvr-distill-model`、`--swiftvr-distill-view-window`（既定 15。3.3 の view 平滑化を再利用）、`--swiftvr-distill-strength`（既定 0.75、0〜2。双線形拡大に足す分の比率）、`--swiftvr-distill-stabilize`（既定 0、実験的。テストクリップでは強さを下げたのと同じ引き換えにしかならなかった）。
- **連動なし**: fp8-recon は自動有効化されず、`--frame-gen` も切られず、SeedVR2 一次とも排他にならず、色補正もありません。GUI には出ません。`--stream` と `--segments` は未検証。
- **状態**: `tests/test_swiftvr_distill.py` + `test_main.py`。Windows（2026-10-05）と Linux（2026-10-08）で検証済み: 1080p 素材で既定値の揺れの増加は +8.6%、質感 122%（二次復元なし基準）、scale 2 の `swiftvr-inline` は +10.9% / 130%。二次段の時間は SwiftVR の約 1/7（1080p 4203 フレーム: 5.8 秒対 38.9 秒）、VRAM の上乗せは約 5.5 GB に対して最大 0.4 GB。8K SBS の VR 素材も完走。利用者の目視では揺れの違和感は少なく質感も悪くない。詳細は [swiftvr.md](swiftvr.md#swiftvr-distill--secondary-restoration-swiftvr-distill)。

### 3.5 RTX Super Res の追加オプション

`nvidia-vfx` 0.2.0.0 で使えるようになった、`rtx-super-res` 二次復元のオプトインの調整 2 点。既定値（strength 1.0、標準モデル）は従来挙動です。

- **`--rtx-strength <0.0〜1.0>`**: SDK の `strength` を RTX の全パス（拡大、ノイズ除去、ブレ除去）に適用。低いほど未処理の画に近づきます。1.0 と異なるときだけ SDK に渡すので、既定構成は 0.1.0.1 の venv でも構築できます。
- **`--rtx-highbitrate`**: 選んだ `--rtx-quality` の水準で `LOW..ULTRA` の代わりに `HIGHBITRATE_{LOW..ULTRA}` の拡大モデル族を使います。圧縮アーティファクト抑制を省くモデルで、BasicVSR++ の出力には圧縮ノイズがないため、ディテールが残るという仮説です。ノイズ除去とブレ除去には影響しません。
- 配線: `SessionConfig.rtx_strength` / `rtx_highbitrate`、`RtxSuperresSecondaryRestorer(strength=, highbitrate=)`、GUI の RTX Super Res パネルのスライダーとスイッチ（video-session キーの一部）、`tests/test_rtx_superres_restorer.py`。
- **状態**: 1080p テストクリップで両方とも出力が測定可能に変わり（既定との PSNR: `--rtx-highbitrate` 47.8 dB、`--rtx-strength 0.5` 48.7 dB）、目視（2026-10-03）でアーティファクトなし。「よりよく見えるか」は素材依存のため、両方ともオプトインのままです。

### 3.6 LTX 出力の鮮明化（`--ltx-crop-vsr`）

このビルドは LTX を走らせられませんが、LTX の scan と合成は公開されています。`--ltx-crop-vsr swiftvr|swiftvr-distill` は、公式リリースが LTX で復元した動画（`--input`）とその元動画（`--ltx-crop-vsr-source`）を受け取り、元動画に LTX の scan を走らせ直し、同じ camera で LTX 出力を切り出し、121 フレームの window ごとに SwiftVR（本家モデルを inline worker で scale 2、または distill をプロセス内）で ×2 して、LTX 自身の `composite_frame` で合成し直します。出力の解像度、フレーム数、タイムスタンプは LTX 出力のままです。CLI 専用、1 本ずつ。

- **モジュール** `jasna/ltx/refine.py`: 整合確認（2 本をパケット単位で demux し、枚数、解像度、各フレームの stream 先頭からの時刻が半フレーム以内で一致すること）、scan（`scan_span` を 1 segment で。`--ltx-large-canvas` は指定どおり。終了後に検出器を解放）、1 パスの駆動部 `WindowRefiner`（LTX フレームが届くたびに canvas を作り、window が揃ったら ×2、覆う window がすべて終わるまでフレームを pinned ホストバッファに保持（最大 121 枚））。LTX 出力と元動画を並走デコードし、track polygon 内の輝度差平均で LTX が復元した window と触っていない window を見分け（閾値 3/255）、後者は警告して skip します。このビルドの検出器数値差だけで出た track を、モザイクのまま鮮明化しないためです。
- **配線**: `SessionConfig.ltx_crop_vsr` / `ltx_crop_vsr_source`。`build_restoration_session` は検出器のエンジンだけコンパイルし一次モデルを読みません。`build_ltx_crop_vsr_restorer` が scan 後に restorer を作り（scale 2、view 平滑化なし）、`Pipeline.run` が `_run_ltx_refine` へ分岐します。BasicVSR++ の重みは不要。
- **拒否**: 一次側のオプション、フレーム列や全画面を変えるもの（`--segments`、`--ltx-auto-split`、`--stream`、`--retarget-high-fps`、`--fmp4`、`--frame-gen`、VR、画像とフォルダ入力、`--lut`、`--sharpen`）、`--swiftvr-scale 4`、view-window 系、`--ltx-large-canvas` と `--swiftvr-distill-stabilize` の併用。
- **状態**（Linux、2026-10-16）: 30 秒のプローブ素材 2 組（720x480 と 1080p、リリース v0.11.1 の LTX 出力）で両方式が完走し、plan はリリースと同じ（2 track、11 window）、枚数とタイミングは LTX 出力と一致（0.5 ms 以内）。再エンコード基準で texture 115〜122%、揺れの増加 +8〜11%。30 秒素材の壁時計は distill で 23〜28 秒（VRAM 上乗せ 2.5〜2.8 GB）、SwiftVR で 50〜55 秒（9.8〜10.8 GB、16 GB カード向け）。ユーザー目視（404p〜720p の実素材 5 本、SwiftVR 強度 1.0）は狙いどおりで合格。Windows（RTX 5060 Ti、2026-10-16）でも同じ 2 組の両方式が完走し、plan、枚数、時刻は Linux と同じ（壁時計は distill 72〜76 秒、SwiftVR 135〜181 秒、SwiftVR の VRAM は GPU 全体で最大 14.1 GB）。768 px canvas、小さいモザイクの素材は未了。テスト: `tests/test_ltx_refine.py`、`test_ltx_refine_cli.py`、`test_ltx_refine_gpu.py`。詳細は [ltx_crop_vsr.md](ltx_crop_vsr.md)。

---

## 4. フレーム生成

AI 補間フレームを挿入して出力フレームレートを上げます: `--frame-gen {none,2x,4x}`。ファイル出力のみ。

- **なぜ二次復元ではなく新ステージか**: 二次復元は 256px クロップを処理しフレーム数を変えませんが、フレーム生成は全解像度の出力フレームに対してフレーム数と PTS を増やします。パイプラインの `FrameWriter` を薄く包むデコレータ（`FrameGenWriter`）で、本体とエンコーダは無改造です。連続する 2 フレームごとに実フレームを出力後 `M-1` 枚の補間を `pts_k = prev_pts + round((curr_pts - prev_pts) * k / M)` で挿入。音声はタイムコードを保つので尺と同期は不変。総フレーム数は `(N-1)*M + 1`。エンコーダの fps/GOP はソース fps × 倍率を `output_fps` に渡して正します（`--retarget-high-fps` の間引き後 fps にも倍率が乗る）。
- **バックエンド**（`--frame-gen-backend`、`FrameGenerator` プロトコルで差し替え可能）:
  - `rife`（既定）: blend-encode スレッド上で全解像度の RIFE を PyTorch 実行、既定 fp16（`--fp16` に追従。float32 グリッドを焼き込んだ外部 TorchScript は検出して fp32 へ）。TorchScript チェックポイント（推奨）か、同梱 RIFE 4.6 `IFNet`（`jasna/models/rife/`）への state_dict を読みます。重みは `--frame-gen-model-path` か `model_weights/rife.pth` で、**同梱しません**（Practical-RIFE の重みは非商用条項）。`scripts/make_rife_torchscript.py` が Practical-RIFE の checkout を変換します（4.25 で確認）。fp16 は fp32 の約 1.9 倍速で両者の PSNR は約 50 dB。
  - `rtx`: nvvfx `VideoFrameGeneration` による NVIDIA RTX Video Frame Generation（`nvidia-vfx` 0.2.0.0 以降）。重み不要。compute capability 8.9 以上（Ada / RTX 40 以降。SDK は Turing/Ampere で動かない）限定で、満たさなければ `rife` を案内するエラー。Linux はドライバ 570.190 / 580.82 / 590.44 以降。`--frame-gen-rtx-mode {low,medium,high}`（既定 `medium`。`high` は約 6 倍遅く改善はわずか）。最初のフレームサイズで遅延ロードし、サイズ変更時に再ロード（約 15 ms）するのでフォルダ一括でも generator を 1 つ共有。RTX 5080 の実測: 1080p で生成 1 枚 1.53 ms 対 RIFE fp16 10.1 ms（2160p は 2.51 対 48.3 ms）、補間 PSNR は RIFE 比 +0.7 dB（1080p）/ +1.6 dB（2160p）、RIFE との目視 A/B 合格（残像やエッジの破綻なし）。Windows も検証済み（2026-10-03）。`jasna/framegen/rtx_frame_generator.py`。
- **スタンドアロン `jasna-framegen`**（`jasna/framegen_cli.py`、2 つ目の console script）: 復元済み動画に**フレーム生成だけ**を適用します（検出も復元もしない）。同じ NVDEC/NVENC 経路と `FrameGenWriter` を再利用し、`jasna.pipeline` と `jasna.protection` を一切 import しません（テストで担保）。公式リリースで復元 → ここでアップコンバート、という 2 パス運用ができます。フォルダ入出力 + `--output-pattern`、動画のみ、バッチ全体で generator を 1 つ共有。`--rtx-mode` で RTX の品質モードを選びます。
- **GUI**: **エンコード**セクションに倍率、バックエンド、RTX モード、RIFE 重みパス。
- **制約**: `--stream` 不可。`--segments`（コピー区間と出力レートが一致しなくなる）、LTX（LTX パスは frame-gen writer を通らずエンコーダへ直接書く。GUI は LTX ジョブでスキップ）、SeedVR2 一次、FlashVSR / SwiftVR の両モード（フレーム生成は別パスで）と併用不可。`swiftvr-distill` は切りません。
- **状態**: `tests/test_frame_gen_writer.py`、`test_rife_frame_generator.py`、`test_framegen_cli_*.py`。両 OS の実機で、30fps / 10661 フレームの入力が 60fps / 21321 フレームになり、`ffprobe` で尺と音声同期の不変を確認。詳細は [frame_generation.md](frame_generation.md)。

---

## 5. メディアとエンジンのバックエンド

### 5.1 torchcodec 動画バックエンド

native（upstream の PyAV NVDEC/NVENC と、その VALI → PyAV-hw → PyAV-sw デコード連鎖）の代わりに使える実験的な `torchcodec>=0.15.0` バックエンド。オプション依存、既定 off。`--video-backend {native,auto,torchcodec}` と個別上書きの `--decode-backend` / `--encode-backend {inherit,...}`。GUI は**エンコード**の「ビデオバックエンド」ドロップダウン。`--video-backend` は upstream の `JASNA_DECODE_BACKEND` 連鎖の**上位層**で、`native` 選択時にその連鎖が働きます。

- **decode**: `auto` は torchcodec を優先し、失敗時は native へフォールバック。`--retarget-high-fps` の frame stride は native 専用（torchcodec 強制時はエラー、`auto` は native へ）。
- **encode**: **`auto` では選択されません。** upstream の native エンコーダは HEVC/AV1 を 10-bit P010 で出力し、torchcodec エンコーダは 8-bit 専用なので、黙って選ぶと出力ビット深度が変わるためです。`--encode-backend torchcodec` の強制時のみ、8-bit ソース + HEVC/AV1 + マッピング可能な NVENC 設定で動きます（streaming、`--segments`、フレーム生成、fps 変更は拒否。`--sharpen` は適格性ゲート、`--fmp4` は警告）。色空間は encoder 内蔵の ffmpeg copy-remux が H.273 コードポイントで付与します（HEVC は `hevc_metadata` BSF の VUI 書き換えも）。
- 起動ログに選択結果（`[decode: ..., encode: ...]`）、リーダー/エンコーダに `backend` 属性、エンコードは専用ワーカースレッド。`jasna/media/backend.py`（選択）、`torchcodec_decoder.py`、`torchcodec_encoder.py`。
- **既知の未対応**: torchcodec エンコーダは章は保存するが字幕・データストリームは脱落（native は全部保存）、CQ 既定値はハードコード 25（upstream の codec 別既定と異なる。`--cq` 明示指定は届く）。
- **状態**: 両 OS で検証済み。設計、能力マトリクス、計測は [torchcodec_backend.md](torchcodec_backend.md)（デコードは中立、律速は検出段）。

### 5.2 cuDNN FP8 upsample バックエンド（`--fp8-recon`）

BasicVSR++ の **upsample** サブエンジンを cuDNN graph API の FP8 畳み込みで置き換える実験的バックエンド。lada-ex `feat/fp8-recon`（AGPL-3.0）からの移植。`--fp8-recon` で opt-in（GUI とサブプロセス経路のため `JASNA_FP8_RECON=1` にブリッジ。GUI は「FP8 復元（実験的）」トグル）、既定 off は従来挙動。FP8 対応 GPU（sm89 以上）、fp16、依存 `nvidia-cudnn-frontend`（cuDNN 9.17 以上は torch cu130 wheel に同梱。Windows は compile glue 用の `triton-windows` と eager フォールバック）が必要で、構築に失敗したら警告して TensorRT エンジンへフォールバックします。

- 有効時は TensorRT upsample エンジンを**ロード自体しない**（`load_sub_engines(..., load_upsample=False)`）ので、ロード時アリーナが確保されません。エンジンファイルはフォールバック先として引き続きビルド・保持。cuDNN グラフのバケットは upstream の固定 `UPSAMPLE_BATCH = 30` 上限で、warmup は `--max-clip-size` に依存せず、全 T バケットを事前ビルドします（jasna の clip 長は自由に分布するため）。
- 現在の主用途は**同時常駐の VRAM 予算**です。`flashvsr-inline` と `swiftvr-inline` はこれを自動有効化し、常駐 worker の隣で一次を ~1.6 GB に絞ります。パイプラインでは 480p〜4K で VRAM ピークが 0.9〜1.7 GB 下がり、e2e fps は不変（検出律速）。ステージ出力は FP32 比 PSNR 64 dB、FP16 エンジンと目視で区別できず、走行間で bit 決定的。
- ここで保持する統合修正: `import cudnn` の前に torch 同梱 cuDNN の lib ディレクトリを `LD_LIBRARY_PATH` の先頭に置きます（nvvfx がサブライブラリ欠落のディスパッチャ入りディレクトリを先頭に追記するため）。同じパス漏れが、外部 worker の環境を掃除する理由でもあります（[7](#7-このブランチで保持しているバグ修正)）。
- `jasna/restorer/fp8_upsample.py` と `basicvsrpp_sub_engines.py` の注入・フォールバック・close。A/B ゲートベンチは `jasna --benchmark --benchmark-filter fp8`。
- **状態**: Linux・Windows で検証済み（両 OS で A/B ベンチの全ゲート通過、1080p フルパイプライン完走）。Windows の frozen build での `cudnn` 同梱は未検証（TRT へフォールバック）。詳細は [fp8_recon.md](fp8_recon.md)。

### 5.3 TensorRT-RTX フレーバー（`nvidia-rtx` extra）

`nvidia` の代わりに `nvidia-rtx` extra を入れると、TensorRT スタック全体（生 API の検出エンジンと torch-tensorrt dynamo の BasicVSR++ サブエンジン）が TensorRT-RTX に切り替わります。狙いは初回の待ち時間で、JIT コンパイルのためコールドビルドが 3〜9 倍速くなります（RTX 5080: RF-DETR v6 36 秒 → 5 秒、BasicVSR++ サブエンジン 6 個 55 秒 → 16 秒。RTX 5060 Ti: 118 秒 → 16 秒 / 143 秒 → 52 秒）。代償: 検出推論が約 12% 遅い（層別プロファイルで mask 出力周りの JIT 融合品質に起因と特定。CUDA graph、特殊化戦略、AOT ビルダー設定、新しい tensorrt-rtx のいずれでも縮まず、差は受け入れた）、長尺の定常スループットが約 −10%、BasicVSR++ エンジンのロード時 JIT が毎プロセス数秒（生 API エンジンにはエンジン隣接のバージョンタグ付き `.jitcache` を持ち、推論後にも再保存して形状特殊化済みカーネルを残す）。標準フレーバーとの出力一致は e2e クリップで PSNR 46.7 dB / SSIM 0.994。

- 実装: フレーバー判定は `engine_paths.trt_flavor()` の `find_spec("tensorrt_rtx")` probe（torch-tensorrt-rtx が張る `sys.modules["tensorrt"]` 別名に影響されない。`JASNA_TRT_FLAVOR` で強制可）。`jasna/trt/_backend.py` を TRT モジュールの唯一の供給点とし（ガードテストあり）、`engine_system_suffix()` に `.rtx` タグを注入して両キャッシュを 1 つの `model_weights` に共存させます。TensorRT-RTX は strongly-typed で `BuilderFlag.FP16` を拒否するため、ONNX をパース前に丸ごと fp16 変換します（`_convert_onnx_bytes_to_fp16`。RF-DETR に焼き込まれた 44 個の `Cast(to=float32)` を含む。省くと黙って fp32 エンジンになり 5 倍遅い）。1 venv に 1 フレーバー。検出エンジンの whole-graph CUDA graph capture は `JASNA_TRT_RUNNER_CUDAGRAPHS` でオプトイン。
- **状態**: Linux・Windows で検証済み。詳細は [tensorrt_rtx.md](tensorrt_rtx.md)。

---

## 6. GUI

### 6.1 A/B モデル比較（区間エディター）

区間エディターから開くモーダルウィンドウで、検出モデル × 復元チェックポイントの 2 組を同一フレーム（または短い再生窓）で左右比較します。モデルの世代交代やファインチューン評価のたびに動画全体を 2 回処理する手間を置き換えます。ビューは **復元** / **検出**（マスクオーバーレイ + スコア、側別の信頼度スライダー）/ **クリップ**（左右同期再生。A 側がクロック）の 3 つ。実行は **実行** ボタンで明示的に行い、選択や時刻の変更は「（未反映）」表示だけを更新します。

- **逐次実行を構造で保証**: torch_tensorrt の CUDA graphs モードはプロセスグローバルで、キャプチャ中に他スレッドが発行する CUDA 呼び出しに汚染されるため、2 レグは単一ワーカースレッド（`jasna/gui/ab_compare_worker.py`）が A→B と順に実行します。レグ間に届いたリクエストは B を打ち切り、両セッション常駐で OOM したら両スロットを解放して現レグのみコールドで 1 回リトライします。
- **パイプラインの流用**: 両レグは upstream 共有の `run_restoration_pass()`（復元プレビューの短窓パスで PIL 画像を回収）を使います。元は modi がこの関数を抽出しましたが、upstream v0.11 が自前の版（`b0e5e5c`）を入れたので、A/B はそれを使います。
- **実行経路の可視化**: BasicVSR++ はサブエンジン欠落時に無言で PyTorch にフォールバックします。`BasicvsrppMosaicRestorer.tensorrt_active` が各側の **TRT** / **PyTorch** バッジを駆動し、エンジンの無いチェックポイントは 15〜60 分のコンパイル subprocess に入らず PyTorch 実行（`compile_basicvsrpp=False`）になり、ウィンドウ内の **エンジンをコンパイル** ボタンで明示的にコンパイルできます（エンジンはチェックポイント単位なので他方の側のバッジも更新）。
- **EMA ガード**: `load_model` は `is_use_ema=True` 固定のため、`generator_ema.*` キーを持たないチェックポイントは無言でランダム重み実行になります。`jasna/restorer/checkpoint_info.py` が実行前に検出（mmap ロード + キー走査、path/mtime/size でキャッシュ）して拒否します。
- **チェックポイントの選択**: GUI の復元モデルパスを `default_restoration_model_path()`（`jasna/engine_paths.py`）に一元化し、`build_video_session` が `restoration_model_path` を受けます。
- **ズーム/パンの共通化**: `jasna/gui/preview_zoom.py`（正規化座標、issue #229 の HiDPI 除算）が 1 つの `ZoomPanController` で両ペインを駆動し、区間エディター本体の数式も同モジュールへ委譲します。
- **SeedVR2 側**: 外部 checkout（`$JASNA_SEEDVR2_REPO`、既定 `~/seedvr2_videoupscaler`）と LoRA があるとチェックポイント選択に `seedvr2` 項目が加わります。EMA 検査、TRT バッジ、コンパイルボタンはその側ではスキップされ、VR 動画は言語別メッセージ付きで拒否されます。
- UI 文字列は 5 ロケールすべてに追加。`jasna/gui/ab_compare.py`、`ab_compare_worker.py`、`preview_zoom.py`、`jasna/restorer/checkpoint_info.py`、起動導線は `segment_editor.py`。
- **状態**: `tests/test_ab_compare_{worker,window}.py`、`test_preview_zoom.py`、`test_checkpoint_info.py`、`test_segment_editor.py` の統合分（CPU のみ。ワーカーの逐次性はテストで固定）。Linux で 720p〜8K VR 60fps 素材、Windows 11 で HiDPI 125%/150%、Tk grab のモーダル制御、コンパイル subprocess、キュー開始ボタンのロックを確認。利用者向け説明は [segments.md](segments.md#ab-モデル比較)。

### 6.2 動画プレーヤーの復元チェックポイント選択

復元動画プレーヤーの設定カード（検出モデル・二次復元・信頼度）に **復元チェックポイント** の選択を追加しました。選択肢は `model_weights/` 直下の `*.pth`（A/B 比較と同じ `discover_restoration_checkpoints()`）、既定は generic v1.2。選択は `AppSettings.restoration_checkpoint`（チェックポイントの stem、空 = 既定）として流れ、`video_session_key()` に含まれるため、切り替えは検出モデル変更と同じ経路でプレーヤーのパイプラインを再構築します。`resolve_restoration_checkpoint()` が stem → パスを解決。ガードは A/B 比較と同じで、EMA 重みの無いチェックポイントはセッション構築前に拒否し、サブエンジン未コンパイルのチェックポイントは PyTorch 実行（`compile_basicvsrpp=False`）になり、ステータスにその旨を表示します（エンジンを作るには CLI の `--restoration-model-path` 実行か A/B ウィンドウ）。v0.11.1 ベースからは `AppSettings.restoration_model` が upstream のモデル選択（BasicVSR++ / LTX）になり、stem をそこに保存していた旧 modi プリセットは読み込み時に移行されます。

### 6.3 modi 機能の設定項目

- **基本**: 「FP8 復元（実験的）」トグル（`--fp8-recon`）。
- **エンコード**: **フレーム生成**（倍率、バックエンド、RTX モード、RIFE 重みパス）と**ビデオバックエンド**（native / auto / torchcodec、両側に適用）。
- **二次復元**: RTX Super Res の**強さ**スライダーと**高ビットレートモデル**スイッチ。
- **区間エディター**: **A/B 比較…** ボタン。**動画プレーヤー**: 復元チェックポイントのドロップダウン。
- GUI に出ないもの: SeedVR2、FlashVSR、SwiftVR、SwiftVR distill、`--ltx-crop-vsr`、TensorRT-RTX フレーバー（venv の選択）。

---

## 7. このブランチで保持しているバグ修正

`modi` で修正し、upstream には無いバグ。upstream が後から自前で直したものは[付録 A](#付録-a-upstream-に吸収されたまたは-drop-した変更)にあります。

- **一次復元のバッファ保持レース**（`jasna/restorer/restoration_pipeline.py`）: TRT runner の永続出力バッファの view を保持したまま別スレッドで後から uint8 へコピーアウトしており、その前に次クリップの推論が同バッファを上書きすると、復元領域が別クリップのデータになっていました。`primary_raw = self.restorer.raw_process(...).clone()` で、バッファが再利用される前に一次 stream 上でコピーアウトします。Linux の「チャンク移動」（issue #2 / upstream #158）のもう一方の原因（デコーダ側）は upstream v0.6.2 の stream 同期が直しており、この復元側の箇所はそれとは独立です。
- **stage の例外で pipeline が失敗せずハングする**: stage スレッドが死ぬと（例: 二次復元が worker エラーで例外）、その生産側が満杯の `FrameQueue.put` で永久に待ち、`Pipeline.run()` が返りませんでした（進捗 100% のまま encode-stall 監視が 30 秒ごとにログ）。`_run_pass` は join 中に `error_holder` を監視し、最初のエラーで cancel を立て、フレームキューを abort（`FrameQueue.abort()` が待機中の `put` を解放）、metadata キューを排出して全 stage を巻き戻し、元の例外を再送出します。upstream v0.11 の `_run_pass` 上に再実装。`test_run_secondary_error_does_not_hang_with_blocked_producer` で固定。
- **フォルダ一括でのフレーム生成**: generator は 1 回だけ構築して全動画で共有しますが、`FrameGenWriter.close()` が 1 本目の後にその借用 generator を close していたため 2 本目がクラッシュしていました。writer は generator に触らず、構築した側（CLI のフォルダループ / GUI のジョブ）が一括処理後に 1 回 close します。
- **外部 worker が jasna のライブラリディレクトリを継承していた**（`jasna/restorer/worker_env.py`、2026-10-16）: FlashVSR、SwiftVR、SeedVR2 の worker は自前の torch と cuDNN で動きますが `LD_LIBRARY_PATH` を継承しており、FP8 が on（`swiftvr-inline` が有効化する）だと新しい torch の worker が jasna 側の cuDNN サブライブラリを読んで最初の畳み込みで死にました（`CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`。SwiftVR venv が torch 2.14 / cuDNN 9.24 に移って顕在化）。`external_worker_env()` が全 worker の環境を作り、`PYTHONPATH` を落とし、`LD_LIBRARY_PATH` からこのインタプリタの配下にある項目を除きます。
- **Linux/X11 でモーダルダイアログが空表示になる**: About ダイアログと初回ウィザードは upstream の `grab_modal()` を 250 ms 遅らせて呼びます（`jasna/gui/app.py`、`jasna/gui/wizard.py`）。ウィジェットの描画より先にモーダルグラブを掛けると、一部の X11 ウィンドウマネージャでリサイズ不可の `CTkToplevel` が空のまま残るためです。upstream の他のダイアログはグラブ前に `wait_visibility()` を呼んでおり、変更していません。
- **FP8 バックエンドの cuDNN ロード順**: [5.2](#52-cudnn-fp8-upsample-バックエンド--fp8-recon) を参照。
- **frozen build 向けの YOLO ONNX CPU エクスポート**: ultralytics の GPU エクスポートは TensorRT より先に torch CUDA を初期化し、frozen バイナリで初期化順序を壊します（CUDA error 100）。検出 ONNX は CPU でエクスポートし（`half=False, device="cpu"`）、upstream の `CUDA_VISIBLE_DEVICES` 退避/復元は維持。エンジンは引き続き fp16 でビルドされます。

---

## 8. ビルド・依存・パッケージング

- **`pyproject.toml` の差分**: `nvidia-vfx==0.2.0.0` を固定（upstream は無 pin。`rtx` フレーム生成バックエンドが 0.2.0.0 で初めて公開された `VideoFrameGeneration` effect を必要とし、無 pin のままでは新規 venv に旧スタブが想定しない API の版が黙って入るため。副作用: 0.2.0.0 は TensorRT を同梱しなくなり旧 `libnvinfer.so.10` 衝突は構造的に消滅、Linux wheel は約 1.1 GB から 443 MB に縮小）。`nvidia-cudnn-frontend` と `triton-windows`（FP8 バックエンド）。**`nvidia-rtx` extra**（torch-tensorrt-rtx 2.12.1、`onnxconverter-common`。1 venv で `nvidia` と排他）。console script `jasna-framegen`。optional の `torchcodec`。GPU スタックのピン自体は upstream に追従（torch 2.12.1+cu130 / torchvision 0.27.1 / torch-tensorrt 2.12.0 / tensorrt 10.16.1.11 / comfy-kitchen 0.2.38 / `av==18.1.0` / `python_vali==4.8.8`）。`python_vali` 4.8.8 は PyPI に無く（PyPI の最新は 4.8.7。2026-10-16 時点）、jasna の VALI デコーダはフォークにしかない `DecodeSingleSurfaceAsyncDetailed` を使う。そのため両 OS のビルドガイドとも、先に Kruk2 氏の VALI フォーク wheel をビルドし、`--find-links` でインストールに渡す（ビルドガイド §5.1〜5.3）。
- **`model_weights/` の自動解決**（`jasna/model_weights_resolver.py`）: 環境変数 `JASNA_MODEL_WEIGHTS_DIR`、パッケージ親フォールバック、ロギング。upstream の frozen バイナリ向け `engine_paths.model_weights_dir()` の呼び出し箇所もここを通ります。
- **Windows の DLL ロード補助**（`jasna/packaging/windows_dll_paths.py`）: torchcodec バックエンド向けの FFmpeg 8 DLL 解決（`CUDA_PATH` とバンドルの `tools\` を参照）。働くのは frozen ビルドだけで、ソース実行では即 return する。
- **同梱パッチ**: `patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff`（venv の mmengine に当てる。Windows・Linux ビルドガイドとも §5.4。upstream は `7c64105` で `checkpoint.py` の 1 行を `weights_only=True` に変えたが、その版ではソース実行で BasicVSR++ の `v1.2.pth` が `HistoryBuffer` を理由に読めないため、フォークは `weights_only=False` のまま同梱する）と `patches/flashvsr_plus_tinylong_multichunk_fix.patch`（upstream の FlashVSR_plus checkout 向け、[3.2](#32-flashvsr-二次復元)）。
- **スクリプト**: `scripts/build_nuitka.py`（フォーク独自の Windows 向け Nuitka バンドル。`jasna.exe` + `jasna-framegen.exe`、RIFE 重みは `--bundle-rife` 指定時のみ同梱、worker/driver スクリプトは実ファイルとして複製。[frozen_build.md](frozen_build.md)）と `scripts/make_rife_torchscript.py`。
- **ビルドガイド**: [building_linux.md](building_linux.md) と [building_windows.md](building_windows.md)（必須の VALI フォーク wheel、RIFE 重み、外部 venv を含むソースからのセットアップ）。両ガイドとも本ベースで検証済み（2026-10-16。Linux は RTX 5080、Windows は RTX 5060 Ti）。同日、両ガイドの 5 節を「ビルド用パッケージ（`setuptools<81` を含む）→ VALI wheel のビルド → `--find-links` 付きの jasna インストール → mmengine パッチ」の順に組み替えた。Windows の VALI ビルドはこの順序の最小構成の venv で確認済み。Linux では空の venv からこの順序で通し、`[nvidia]` の解決、フォーク API の存在、mmengine パッチ後の `v1.2.pth` 読み込みまで確認した（2026-10-16）。
- **整合のため残すもの**: `_preload_tensorrt_runtime()` は無害な no-op として残します（対象だった nvvfx 同梱 TensorRT の衝突は 0.2.0.0 で消滅）。
- **既知の未対応**: torchcodec エンコーダのストリームと CQ の差（[5.1](#51-torchcodec-動画バックエンド)）。**オフライン**の `flashvsr` / `swiftvr` は動画単位の `--post-export-video-command` を実行しない（キュー単位の `--post-export-action` は実行される。inline は通常パイプラインなので正常）。FP8 と VALI フォーク wheel は frozen バンドル内では未検証。

---

## 9. 機能の組み合わせ

✓ 併用可（起動時に拒否されない）、✗ 起動時に拒否、warn = 警告して実行、auto = 自動で有効化、— 該当なし（LTX 列は、LTX を走らせる必要がある箇所はこのビルドで実行できないため —）。

| 機能 | `--stream` | `--segments` | `--frame-gen` | VR（`--vr-mode`） | フォルダ / 画像入力 | LTX 一次 | SeedVR2 一次 | inline の FlashVSR / SwiftVR | `--fp8-recon` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SeedVR2 一次 | warn | ✓ | ✗ | ✗ | ✓ | — | — | ✗ | ✓ |
| FlashVSR オフライン（`flashvsr`） | ✗ | ✗ | ✗ | ✗ | 動画のみ | — | ✓ | — | ✓ |
| FlashVSR inline | ✗ | ✓ | ✗ | ✓ | 動画のみ | — | ✗ | — | auto |
| SwiftVR オフライン（`swiftvr`） | ✗ | ✗ | ✗ | ✗ | ✗ | — | ✓ | — | ✓ |
| SwiftVR inline | ✗ | ✓ | ✗ | ✓ | 動画のみ | — | ✗ | — | auto |
| SwiftVR distill | 未検証 | 未検証 | ✓ | ✓ | ✓ | — | ✓ | — | ✓ |
| `--ltx-crop-vsr` | ✗ | ✗ | ✗ | ✗ | ✗ | — | ✗ | swiftvr バックエンドのみ | — |
| フレーム生成 | ✗ | ✗ | — | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| torchcodec encode（強制） | ✗ | ✗ | ✗ | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| torchcodec decode | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| `--fp8-recon` | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | auto | — |
| TensorRT-RTX フレーバー | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ |

補足: オフラインの各モードは `--retarget-high-fps` も拒否。torchcodec decode は `--retarget-high-fps` ではスキップ。`--fp8-recon` は `--fp16` と sm89 以上の GPU が必要。`rtx` フレーム生成バックエンドは compute capability 8.9 以上。A/B 比較は seedvr2 側で VR 動画を拒否。

---

## 付録 A: upstream に吸収された、または drop した変更

もはや本ブランチの差分ではない旧 modi 変更です。最終列のリベースで除き、以後は upstream の実装に従っています。

| 旧 modi 変更 | upstream 側の対応 | 除いたリベース |
| --- | --- | --- |
| ブレンドマスク生成の分離 box blur（密な `K×K` 畳み込みを 1D 2 パスに） | 同等実装 `f8c4048`「Reapply separable conv」 | v0.6.2 |
| デコーダの `torch.cuda.synchronize()` バリアと opt-in の `--decoder nvdec`（Linux の「チャンク移動」、issue #2 / upstream #158） | `cf33bcf`「sync decoder cuda stream」がデコーダの stream 同期を再導入。復元側のレースはここで修正を保持（[7](#7-このブランチで保持しているバグ修正)） | v0.6.2 |
| torch 2.12 スタック、BT.601 色空間、`onnx` 依存 | upstream が独立に実装 | v0.6.2 |
| `_preload_tensorrt_runtime` の名前と YOLO export の `CUDA_VISIBLE_DEVICES` 退避/復元 | upstream `6545b78` / `5b3ca34` がブランチの修正と収束。名前は upstream、CPU export は維持（[7](#7-このブランチで保持しているバグ修正)） | v0.6.2 直後 |
| フォルダ毎ファイルバナー `[i/N] Processing <in> -> <out>` | `2f3a87e` が画像+動画合算カウンタ付きの同じバナーを追加 | v0.7.2 |
| 出力フォーマットの柔軟化: `--codec` / `--bit-depth`、`chw_rgb_to_surface`、BT.601/2020 タグ付け、mkvmerge+ffmpeg remux、`docs/CODECS_AND_COLORSPACE_*` | PyAV エンコーダが `--codec {hevc,h264,av1}`、BT.601/709/2020 × full/limited × NV12/P010 一式、H.273 のインライン付与を実装し mkvmerge を廃止。`--bit-depth` は廃止（hevc/av1 → 10-bit、h264 → 8-bit） | v0.8.0 |
| `python_vali` / `PyNvVideoCodec` 向け Windows DLL ロード補助 | PyAV メディア層がそれらをランタイムから撤去 | v0.8.0 |
| NVENC 入力ピッチ整列（256 バイトのステージングバッファ。[#230](https://github.com/Kruk2/jasna/issues/230) として報告） | `bb6e36e`「fix(encoder): align NVENC input pitch」（16 バイト整列、ゼロ埋め、テスト） | v0.8.1 |
| Linux ドライバ最小 580 / `os_utils` のプラットフォーム分岐 | `041eded` | v0.8.1 |
| `--mp4-fast-start`（改名後 `--fmp4`）と StreamingEncoder の broken-pipe 修正 | 本フォークの提案 [#244](https://github.com/Kruk2/jasna/issues/244) から `cda1068` / `229403f` として採用 | v0.9.0 |
| NVENC 非対応オプション組み合わせの黙殺修正 | `af9374b` | d7a99bd |
| VR 空間メタデータ注入と `--mp4-fast-start` の VR180 フォールバック | upstream が `media/spatial_metadata.py` を削除 | d7a99bd |
| TRT 静的 shape bind / YOLO チャンク分割の OOB 修正（illegal access / Xid 31） | dynamic-batch エンジンと `TrtRunner` の self-pad、`c662ed5` + `c79fb88` | v0.9.0 |
| VR fisheye の mask 合成ゴースト修正 | 対象の全フレーム fisheye warp が領域単位の投影再設計で消滅 | v0.9.0 |
| RTX Super Res の TensorRT 衝突に対する `libnvinfer.so.10` 先読み | `nvidia-vfx` 0.2.0.0 が TensorRT を同梱しなくなり構造的に消滅。no-op として残す | 2026-10-03 |
| 復元プレビューからの `run_restoration_pass()` 抽出（A/B 比較と共用） | upstream 自前の共有関数 `b0e5e5c`「run every restoration pass through one function」。A/B はそれを使う | v0.11.1 |
| ビルドガイドに記載していた in-code の `DECODE_BACKEND` トグル | upstream はデコードバックエンドを `JASNA_DECODE_BACKEND` だけで選ぶ（`a3429ecb`） | v0.11.1 |

---

## 付録 B: リベース履歴

各ベースへのリベースで取り込んだ upstream の主変更と本フォークの対応の記録（新しい順）。

### `v0.11.2` ベース（2026-10-18 取り込み）

upstream は v0.11.2 をリリースした（`2fc3e899` から 23 コミット、うち 13 は protection の pin 更新）。主要点: **LTX の less-overlap モード**（`--experimental-ltx-less-overlap`、既定 ON。121 フレームの window の送り幅が 88 から 104 になり、隣接 window が共有する latent は 4 枚から 2 枚、長い track では window 数が約 13% 減る。`--no-experimental-ltx-less-overlap` で v0.11.1 の出力に戻る。`SessionConfig.ltx_window_hop` が必須フィールドとして増え、`plan_video()` が `window_hop` を取る）、**LTX ストリーム重みの正確なサイズでの page-lock**（torch の 2 の冪に切り上げる pinned allocator をやめて `cuMemHostRegister` を直接使う。Windows は `RAM/2 − 2 GiB` の予算内で page-lock し、超えた分は警告を出して pageable RAM で続行するので、8 GB GPU + 16 GB RAM でも LTX が load できる）、**comfy-kitchen 0.2.38**（vendored SageAttention を再取り込み。カーネルの変更は SM 8.9 未満のみで、Ada 以降は bit 一致）、起動時の **HDD インストール警告**（Linux は sysfs の `rotational`、Windows は `IOCTL_STORAGE_QUERY_PROPERTY`。判定できないときは黙る）、INT8 attention のロード失敗理由の表示（#510）、LTX worker の停止タイムアウトでなく本当の失敗を報告（#513、protection）、README と About からの「JAV」の削除、activation における「新しい PC」の定義。`protection` のポインタは `64f4a12c` から `c084d22e` へ（当方では従来どおり未初期化）。

fork 側: modi の 134 コミットを `b623466f` の上に 1 回で再生した。衝突はバージョン行 2 箇所と `gui/video_session.py` の import ブロック（modi が並べ替えた import の隣に upstream が `window_hop` の import を追加）だけ。追従は 2 箇所: `--ltx-crop-vsr` の scan は upstream の `scan_span` を呼び、これが `window_hop` を必須にしたので、`scan_source()` / `run_refine()` が hop を受け取り、pipeline が `SessionConfig.ltx_window_hop` を渡す（GPU 専用の CLI テストで発覚。転送を固定する CPU テストを追加）。`tests/test_ltx_refine.py` の plan は `window_hop=WINDOW_HOP`（88）で作り、期待値を変えない。したがって refine はコマンドラインで選んだ hop に従う（[ltx_crop_vsr.md](ltx_crop_vsr.md) 参照）。エンジンの再コンパイルも VALI の再ビルドも不要で、venv は `comfy-kitchen` を 0.2.36 から 0.2.38 に上げるだけ。検証（Linux、RTX 5080、2026-10-18）: モデル重みのある checkout での全テストを、同じ deselect と timeout で素の upstream `b623466f` と比較し、失敗は upstream と同じ箇所だけ（protection、gcc-15 での native LTX ビルド、表示環境、ultralytics）。10 秒のテストクリップで modi 機能の CLI 実走（BasicVSR++ の FP8 あり/なし、torchcodec、fMP4、RTX Super Res、SwiftVR distill、SwiftVR offline、SeedVR2、RIFE と RTX のフレーム生成、`--segments`、30 秒プローブ素材での `--ltx-crop-vsr swiftvr-distill`、両方の hop での `--ltx-trial`、LTX + frame-gen の拒否）。HDD 判定は本機（NVMe）で False。`comfy-kitchen` 0.2.38 を既存 venv に入れ、kitchen 関連テスト合格。既知の罠: `model_weights/` の**無い** checkout で全テストを流すと `tests/test_fp8_upsample.py::test_fp8_forward_parity_small` で pytest プロセスごと abort する（`cudnn.create_handle()` が abort、exit 134）。v0.11.1 ベースでも同じで新規ではなく、単体では再現しない。

### `v0.11.1` ベース（2026-10-16 取り込み）

upstream は v0.11.0 と v0.11.1 をリリースした（v0.10.0 から 382 コミット）。主要点: **LTX-2.5 拡散復元モデル**（`--restoration-model-name ltx` / `ltx-undistilled`。scan・prepare・denoise・compose の 4 パス、ライセンス推論は worker プロセス、comfy-kitchen の INT8 SageAttention を vendored した native C++/CUDA transformer block（SM 7.5〜12.0 で qualified）、`--ltx-fast`、`--ltx-large-canvas`、`--ltx-seed`、`--ltx-trial`、`--ltx-auto-split`、区間エディターの range 単位のモデルと seed、「大きなモザイクだけ LTX」、GUI からのモデルダウンロード（hf-mirror フォールバック）、残り時間の学習型 speed profile）、**支援者の有効化フロー更新**と**クラウド GPU ガイド**（RunPod、Vast、Docker）、中国向け閑魚の販売者、**README の「はじめに / 上級者向け」分割**と en/ja/zh 全ユーザードキュメントのやさしい英語への書き直し、韓国語・タイ語訳の完成、そして LTX 前の**コアリファクタの波**: `SessionConfig` 全フィールド必須化と `Pipeline` のそこからの構築、`run_restoration_pass()` の一本化、`NvidiaVideoReader/Encoder` → `VideoReader/Encoder`、パッケージ再エクスポートと遅延 alias の廃止、デコードバックエンドの `JASNA_DECODE_BACKEND` 一本化、BasicVSR++ のエンジンコンパイルを `basicvsrpp_sub_engines` へ統合、GUI の worker コールバックをメインスレッドの poller 経由に、区間エディターのセクション分割、v0.8.0 プリセット移行の削除。メディア修正: VC-1 WMV のタイムスタンプ、AAC の二重 ADTS ヘッダ、`hvc1`/`hev1` タグ、DTS の nudging、コンテナが収容できない音声の再エンコード、VALI のスリープ中ディスク待ち。ビルド: torch 2.12.1 / torchvision 0.27.1、フォークビルドでなく PyPI の PyAV 18.1.0、ROCm 10.0、`comfy-kitchen==0.2.36`（`--ltx-trial` にも必須）、`ultralytics>=8.4.96`（onnxruntime 無しの YOLO export、Windows の安全なチェックポイント読み込み）。ほか post-export 修正（Windows のプレースホルダ引用、Linux の `systemctl poweroff`）、Windows の CJK/タイ語 UI フォント、ログパネルの上限、キューのページング。

fork 側: リベースは 2 段階で、段階 1 は `19af9ed`（リファクタの波の末尾。タグ `modi-v0111-stage1`）、段階 2 は `2fc3e899` へ、modi の 118 コミットを再生した。upstream が LTX 期に書き直したファイル（main、session config/factory、GUI の session/processor、locales、tests）は最後に 1 回の 3-way merge で仕上げた。適合: modi の `run_restoration_pass` 抽出は upstream の `b0e5e5c` のために **drop**。stage 例外ハングの修正は upstream の `_run_pass` 上に再実装。**SeedVR2 を upstream の復元モデル選択に統合**（`CLI_RESTORATION_MODELS`、`_DEFAULT_RESTORATION_MODELS` の既定 LoRA、`provide_restoration_models` が BasicVSR++ のスロットに構築）。`AppSettings.restoration_model` は upstream のモデル選択になり、modi プレーヤーのチェックポイント stem は `restoration_checkpoint` へ移動しプリセットを移行。フレーム生成は CLI では LTX と併用拒否、GUI では LTX ジョブでスキップ。エンジン off は `compile_basicvsrpp=False` 経由に（`disable_basicvsrpp_tensorrt` kwarg は廃止）。upstream の改名に追随（`UnsupportedVideoError`、`frame_image`、`default_restoration_model_path(model_name)`、`SegmentEditorState` の新必須フィールド）。`fp8_upsample_ab` は PSNR/SSIM をローカルで計算（vendored mmagic の metric モジュールが削除されたため）。`nvidia-vfx 0.2.0.0` を upstream の `comfy-kitchen` の隣に固定。`.md` の衝突は upstream 側を採ったため、共有ドキュメントへの modi 追記（`cli.md` の行、`gui.md` のプレーヤー注記、`models.md` の RTX 注記、en/zh の `segments.md` の A/B 節）は本書の再構成と同時に再追加した。検証（Linux、RTX 5080）: 同じ deselect と timeout で素の upstream と比較した full suite は modi 3442 passed / 45 failed / 86 errors、upstream 2827 / 41 / 86 で、残る失敗はすべて upstream でも失敗する（protection 依存、native block のビルド、表示環境、ultralytics 版）。modi 機能の e2e CLI 実走は完走し、見つかった 1 件は `worker_env.py` として修正（[7](#7-このブランチで保持しているバグ修正)）。venv は torch 2.12.1 / torchvision 0.27.1 / ultralytics 8.4.178 へ更新。VALI フォーク wheel は Kruk2 upstream/main `0e5c01e`（4.8.8 + ASF/VC-1 タイムスタンプ修正 + decoder-open の GIL 解放）で再ビルドし、両デコードパスが VALI を通り出力は PyAV-hw と md5 一致。エンジンの再コンパイルは不要（エンジン層は無変更）。検証（Windows 11、RTX 5060 Ti、2026-10-16）: full suite は modi 3449 passed / 40 failed / 86 errors、素の upstream 2798 / 38 / 86 で、modi だけの失敗はテスト側の不備 2 件（`ltx_crop_vsr` 属性を欠く Pipeline の組み立てと、子プロセスへ漏れた `JASNA_MAIN_PID`）だったので修正した。VALI フォーク wheel `0e5c01e` は Windows でもビルドでき、両デコードパスの出力は md5 一致、既存の `.win` エンジンはそのまま使われた。modi 機能の CLI 実走 15 系統、`--ltx-crop-vsr` 4 本、GUI（v0.10.0 期プリセットの移行を含む）が合格。外部 worker（SwiftVR、FlashVSR、SeedVR2）の起動に DLL の問題はなく、`worker_env.py` の Windows 版の対策は不要だった。

### `v0.10.0` ベース（2026-08-13 取り込み）

上流 v0.10.0 の主要点: **VLC ベースの復元動画プレイヤー**（`gui/video_player.py` / `gui/raw_player.py` 新設。出力ファイルを作らず復元フレームを全画面再生・シーク。`python-vlc==3.0.21203` がコア依存に追加され、音声再生には 64bit の libVLC 3 = VLC 本体が別途必要。音声なし素材は VLC なしでも SoftwareClock で再生可）、キューの右クリック操作、**動画単位の post-export コマンド**（`--post-export-video-command`、`{input}`/`{output}` 等のプレースホルダ）、**CQ 値のリテラル化**（`media/encoder_quality.py` 新設、`--cq` フラグ、ソース連動ビットレート上限は低ビットレート素材で低 CQ を頭打ちにする仕様）、**章・字幕・データストリームの保存**（`media/container_utils.py` 新設、`_setup_audio` 系は `_setup_source_streams` 系にリネーム）、**`JASNA_DECODE_BACKEND` 環境変数**（`auto`/`vali`/`pyav-hw`/`pyav-sw`。modi の `--video-backend` の下位層で、`native` 選択時にこの連鎖が働く直交レイヤ）、フレームレートの**近接ロック retarget**（±0.5%、59.94fps → 29.97fps を実測確認）、streaming writer 再起動のデッドロック修正、TVAI denoise オプション、AMD 対応強化（AMF smart rendering 等）。**エンジン系（`trt/`、`mosaic/`、restorer の TRT 系）と GPU スタックのピンは無変更**で、既存エンジンキャッシュはそのまま有効。modi 側の適合: `nvidia-rtx` extra の `python_vali` ピンを上流 `nvidia` extra と同じ 4.8.8 に統一。ここで見つかった既知の未対応（現在も残る。[8](#8-ビルド依存パッケージング)）: torchcodec エンコーダは字幕・データストリームを脱落させ CQ 既定 25 のまま、オフライン FlashVSR は動画単位の `--post-export-video-command` を実行しない。

### `v0.9.1` ベース（2026-07-30 取り込み）

上流は v0.9.0 と v0.9.1 を連続リリースした（計 88 コミット）。主要点: **rfdetr-v6 世代の検出器**（デフォルト `rfdetr-v6` @576px/閾値 0.35、4K 向け `rfdetr-v6-large`、VR 向け `rfdetr-vr-v1` + スタジオ別ルーティング）と **dynamic-batch TensorRT エンジン**（`.bs1-N` 命名。固定バッチ機は `TrtRunner` 内で self-pad）、**VR 投影の全面再設計**（新設 `vr_projection.py` によるモザイク領域単位の raw/fisheye/gnomonic conditioning。全フレーム fisheye warp は消滅）、**CUDA カーネル融合 4 種**（RGB→NV12/P010、bilateral denoise、`.cube` LUT、検出前処理。ビルド済み `.fatbin` を同梱）、**VALI NVDEC デコードバックエンドの復活**（VALI → PyAV-hw → PyAV-sw のエスカレーション。Kruk2 氏の corruption-tolerant な `python_vali` フォーク wheel が前提で、素の PyPI wheel では PyAV にフォールバック）、**`--sharpen`**（エンコード前 CAS）、**`--scene-detection`**、ソース連動の**ビットレート上限**、pts 単調クランプ、そして**本フォークの提案（[#244](https://github.com/Kruk2/jasna/issues/244)）から採用された `--fmp4` と StreamingEncoder の broken-pipe 修正**（upstream `cda1068`/`229403f`）。v0.9.1 ではさらに **preprocess/upsample sub-engines の固定バッチ化**（`preprocess_b60` / `upsample_dyn_b30`。`--max-clip-size` 変更での再コンパイルが不要になり、ベース VRAM は解像度に応じて 1.5〜3.5GB 減）、**loop body の CUDA graphs 既定 ON**（kill switch `JASNA_TRT_CUDAGRAPHS=0`）、色変換のゼロアロケーション化（`media/yuv_scratch.py`）、Windows 静的 DPI scaling（`customtkinter==6.0.0` 固定）。本フォーク側では 3 系統を**上流実装採用のため drop**（付録 A）: fmp4 実装一式 + streaming 修正、TRT 静的 shape bind / YOLO チャンク分割の OOB 修正、VR fisheye の mask 合成ゴースト修正。適合対応: `--video-backend` は上流のデコード連鎖の**上位層**として再定義（`native` = 上流の VALI→PyAV エスカレーション。`torchcodec` は従来どおり）、`make_video_encoder` は `--sharpen`/`--fmp4` をパススルー（torchcodec エンコーダでは CAS を適格性ゲート、fmp4 を警告）、**`--fp8-recon` は固定バッチ upsample 前提に再構成**（cuDNN グラフのバケットはクリップサイズでなく固定 `UPSAMPLE_BATCH=30` 上限に）。当時は PyAV wheel を PyAV main から再ビルドする必要があった（エンコーダが `CudaContext(cuda_stream=...)` で CUDA stream を渡すため。後に PyPI 18.1.0 で不要化）。TRT エンジンは初回に全再コンパイル。

### `d7a99bd` ベース（2026-07-23 取り込み）

上流は 4 つの構造リファクタを行った — ①**ドキュメントの per-language 化**（`docs/{en,ja,zh}/` にトピック別ファイル、README から旧セクションを移設）、②**GUI ロケールの言語別分割**（`gui/locales/{en,zh,ja,ko,th}.py`）と **CLI ヘルプ共有テーブル**（`jasna/cli_help.py`、GUI ツールチップが参照）、③**CLI/GUI 共通の composition root**（`session_config.py` / `session_factory.py`）、④**SettingsPanel のセクションモジュール分割**（`gui/settings_sections/`）。加えて **VR 空間メタデータ注入の削除**（`media/spatial_metadata.py` ごと撤去 — 本フォークは追従し、旧 `--mp4-fast-start` の VR180 フォールバックも不要化）、vendored mmagic の inference-only 化、`patch_frozen_torch` のエントリポイント移動、**時間方向の検出フィルタ**（`--max-detection-gap` / `--min-detection-duration`）、RTX Super-Res の入力サイズ設定、AV1 の NVDEC デコード修正、セグメント用ワーキングディレクトリ設定。本フォークの対応: **`--mp4-fast-start` を `--fmp4` に改名**（後方互換なし。preset キーも `fmp4`）。NVENC 非対応オプションの黙殺修正は upstream `af9374b` に取り込まれたため **drop**。modi の GUI 追加（フレーム生成、動画バックエンド、FP8、fmp4）はセクションモジュールへ移植し、flashvsr-inline は `SessionConfig` 拡張 + `session_factory._build_secondary_restorer` 分岐として、フレーム生成/動画バックエンドは `build_pipeline` のパススルー引数として composition root に載せ替えた。modi 独自ドキュメントも `docs/{en,ja}/` の小文字トピック名へ再配置（zh 版は作らない）。

### `v0.8.1` ベース（2026-07-19 取り込み）

上流は **AMD ROCm 対応（実験的、Linux）** を追加した — `jasna/accelerator.py`（vendor/capabilities 判定）、検出の onnxruntime-migraphx 実行（`mosaic/migraphx_runner.py`）、AMF ハードウェアデコード/エンコード、ROCm 向け cumsum ベース box blur（`_prefix_box_blur`）。それに伴い **pyproject の依存が extras に分割**され（`nvidia` = torch cu130 + TensorRT + nvidia-vfx / `amd` = torch + migraphx）、ソース実行の dev インストールは `uv pip install -e .[dev,nvidia]` になった。`requires-python` は **>=3.12** に緩和。本フォークが報告した **NVENC 入力ピッチ整列バグ（[Kruk2/jasna#230](https://github.com/Kruk2/jasna/issues/230)）は upstream `bb6e36e` で修正**され、Linux ドライバ最小 580 化も upstream `041eded` が実装したため、対応するローカル 2 コミットは**このリベースで drop し upstream 版を採用**した（付録 A）。ほか GUI 区間プレビューの HiDPI 修正（#229）、リリースビルド整備（分割 Linux アーカイブ、libnvJitLink.so.13 同梱、Windows AMD リリース）。

### `v0.8.0` ベース（2026-07-18 取り込み）

上流はメディア層を **PyAV（NVDEC/NVENC）へ全面移行**し（`python_vali` / `PyNvVideoCodec` をランタイムから撤去、**mkvmerge 依存も削除**）、**AV1/H264 エンコード**・**BT.601/709/2020 × full/limited × 8/10-bit** の色空間/ビット深度対応・**ソフトウェアデコードフォールバック**・**アナモルフィック（SAR）対応**を実装した。さらに**区間エディター + スマートレンダリング**（`--segments`）、**復元プレビュー再生**、**マスク提案エディタ**（明示的な Submit 時のみ、フレーム+マスクを暗号化して上流の Cloudflare Worker へ送信）、**VR180 復元**（`--vr-mode`）、**高 FPS リターゲット**（`--retarget-high-fps`、60→30 間引き）、多数の Linux GUI 修正が入った。ドライバ要件は Windows 610+ / Linux 580+。ここで初めてリベース方針を明文化: **重複機能は upstream 優先**。本フォークの「柔軟な出力」は upstream 実装に全面置換して drop（`--bit-depth` フラグも廃止。付録 A）、torchcodec バックエンドは新メディア層に合わせて意味論を再定義した（[5.1](#51-torchcodec-動画バックエンド)）。

### `v0.7.2` ベース（2026-06-18 取り込み）

**ベースを `v0.7.2` へ更新。** 以下のコミット別の突き合わせ記録は当初の `v0.6.2` 期リベース向けに書かれたものだが、それらの upstream 修正（デコーダ stream 同期、分離畳み込み、validate-model-name、onnx export、trt load）は現在のベースにも引き継がれており（upstream は v0.7.0 リリース時に `main` を force-push して SHA を書き換えたが、機能は維持）、収束関係は今も成立する。`v0.7.1` ベースで既に入っていたのは upstream の **サポーター向けモデル**（SD 1.5 画像復元、unet-4x）とモデル暗号化/Nuitka 周りで、本フォークはそのコードを **inert（動作しない形）** のまま同梱する（[1](#1-この公開ビルドの範囲)）。ビルドは upstream に合わせて PyInstaller から Nuitka へ移行したが、パッケージングツールは非公開のため、公開での利用経路はソースからの実行。

**新規分:** 上流の**エクスポート後アクション**（`--post-export-action shutdown|command`, `--post-export-command`）、**検出モデルレジストリ / モデルファクトリ**（`jasna/mosaic/detection_registry.py`）、**出力パターン**のファイル名テンプレート（`--output-pattern "{original}_restored.mp4"`）、CLI/GUI のフォルダ処理改善（画像+動画を合算した `[i/N]` カウンタ）、起動時間の最適化（`restorer/__init__.py` の遅延 import）、GUI/Windows 修正。サポーターの `jasna/protection` サブモジュールのポインタも追従（公開フォークでは空のまま）。modi 独自の CLI フォルダ毎ファイルバナーは上流に**吸収された**（`2f3a87e`）ため drop（付録 A）。

### `v0.6.2` 期の突き合わせ（初期リベース）

upstream に取り込まれていた変更はこのリベースで drop した: **分離畳み込みブラー**（upstream `f8c4048`）、**torch 2.12 スタック**、そして `v0.6.2` で新たに入った **issue #2 のカクつきを直すデコーダの stream 同期**（`cf33bcf`「sync decoder cuda stream」）。これにより、本ブランチ独自のデコーダ同期バリアと opt-in の `--decoder nvdec` バックエンドは**冗長になったため削除**し、別系統の一次復元のバッファ保持修正のみを残した（[7](#7-このブランチで保持しているバグ修正)）。**BT.601 カラースペース**も upstream が独自実装したため突き合わせ済み。upstream `v0.6.2` は **HAGS チェックを削除**し、**`onnx` を依存に追加**している。

`v0.6.2` 後の upstream 3 コミット（2026-06-09 取込）も同様に突き合わせ済み: `b55b501`「validate model name」は modi 未改変ファイルのためそのまま取込。`5b3ca34`「ultralytics onnx export fix」（`CUDA_VISIBLE_DEVICES` の退避/復元）は本ブランチの YOLO CPU export 修正と収束。upstream の退避/復元を採用しつつ、frozen build 対策の `half=False, device="cpu"` は本ブランチ側を維持。`6545b78`「linux trt load fix」（nvvfx より先に pip `tensorrt_libs` をプリロード）は本ブランチの既存修正と収束。関数名は upstream の `_preload_tensorrt_runtime` を採用し、実装は本ブランチの防御版を維持。upstream 追加のテストは両ファイルとも modi 実装に対してそのまま通る。
