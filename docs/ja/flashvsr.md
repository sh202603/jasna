# FlashVSR 二次復元(`+modi`)

`--secondary-restoration flashvsr` / `flashvsr-inline` は、一次復元された 256px の
モザイククロップを [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
(one-step streaming diffusion VSR。jasna は
[`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus) fork を使用)
で拡大し、一次の BasicVSR++ では大きなモザイク領域・接写・4K 素材でぼやけがちな
テクスチャの写実性を補う。処理解像度はモデルネイティブの 1024px(4x)か、
`--flashvsr-scale 2` で 512px。どちらでもブレンドがクロップを元の領域へ縮小合成する
ため、出力動画の解像度は変わらない。出力クロップは常に、元になった一次復元結果を
参照して色補正される(「[色補正](#色補正)」)。

FlashVSR には 2 つのモードがあり、どちらもサポートされる。構成で選ぶ:

| | `flashvsr-inline`(単一パス) | `flashvsr`(オフライン 3 段) |
|---|---|---|
| 向く構成 | `basicvsrpp` 一次 + 16 GB カード。単一パスで**中間ファイル・ディスクゲート・二重 encode が無い** | SeedVR2 一次との併用(最高品質構成)、12 GB 級 GPU、段階再開が要る長尺 |
| FlashVSR パイプライン | tiny-long(VRAM がクリップ長に依存しない。推奨 fork はそのまま使える。上流の checkout は**パッチ必須**) | tiny(パッチ不要) |
| `--restoration-model-name seedvr2` との併用 | 起動時エラー(常駐 worker 2 つで 16 GB 超過) | 可 |

以下は主にオフラインモードの説明で、inline は末尾の「inline モード」で扱う。

オフライン 3 段が存在する理由: FlashVSR の tiny モードは**単体で 12–16 GB VRAM** を
消費するため、jasna の一次パイプラインと 16 GB カード上で同時常駐できない。ピーク
VRAM が時間的に重ならないよう処理をプロセス分割することで初めて収まる。inline モードは
FlashVSR の **tiny-long**(定メモリ ~11.9 GB)を使い、一次(fp8-recon で
~1.6 GB)と同時常駐させることで単一パスを実現する。

## 仕組み — オフライン 3 段

`--secondary-restoration flashvsr` の 1 コマンドが 3 つのサブプロセスを順に実行する。
各段は次段が始まる前に完了し、プロセス終了時に VRAM を全解放するため、ピーク VRAM が
同時に存在することはない:

| 段 | 環境 | ~VRAM | 内容 |
|----|------|-------|------|
| 1 (dump) | jasna | ~9 GB | decode + detect + BasicVSR++ 一次復元。各 clip の 256px クロップ + マスク + 幾何をディスク上の **bundle** へ直列化。blend/encode は捨てる。 |
| 2 (FlashVSR) | FlashVSR | 12–16 GB | 各 clip の 256px クロップを 1024px(`--flashvsr-scale 2` なら 512px)に拡大し、色補正して bundle へ書き戻す。 |
| 3 (reblend) | jasna | 軽い | source を再デコードし、bundle から復元結果を再構成、拡大クロップを再 blend して最終出力を encode。 |

Phase 1 / Phase 3 は `jasna --flashvsr-phase {dump,reblend}` のサブプロセスとして
走る(`jasna/__main__.py` で multiprocessing ガードより前に分岐。`--compile-engines`
と同じ流儀)。Phase 2 は `jasna/restorer/flashvsr_phase2_driver.py` を FlashVSR
仮想環境の Python で実行する(jasna を import しない独立スクリプト)。

**bundle** は numpy/JSON ファイルのディレクトリ(`manifest.json`、clip ごとの
`clip_<track>_<start>.npz` と Phase 2 が書く `_fvsr.npz`)。`--flashvsr-bundle-dir`
を指定すると永続化され、途中で失敗した実行を失敗した段から再開できる(完了済み
clip はスキップ)。

blend に必要な幾何(`scale_offsets`)は blend 時に復元フレームの実寸から導出される
ので、FlashVSR の出力はどちらの scale でも**メタデータ改変ゼロ**で再 blend できる。

## 必要なもの

FlashVSR は**同梱していない**。
FlashVSR の checkout・重み・専用仮想環境を利用者が用意し、`--flashvsr-repo` で jasna に渡す。

checkout には fork [`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus)
(既定ブランチ `modi`)を使う。上流の
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) に次の変更を
加えたもので、jasna はこの fork で検証している:

- inline が使う tiny-long のマルチチャンク修正を含む(パッチ不要)。
- `--flashvsr-accel` の高速化(「[高速化](#高速化--flashvsr-accel)」)を持つ。
- `uv sync` 1 コマンドで、`uv.lock` に固定した版の依存が入る。

上流の checkout も使えるが、高速化は使えず、inline にはパッチが要る
(「[上流の checkout を使う場合](#上流の-checkout-を使う場合)」)。

### FlashVSR checkout のセットアップ(一度だけ)

依存は `uv.lock` で固定されており、torch 2.13.0+cu130 / triton 3.7.1(Windows は
triton-windows)/ nvidia-cudnn-frontend 1.29.0 になる:

```bash
# 1. fork を clone(既定ブランチ modi)。models/posi_prompt.pth もこれで入る
#    (repo に git-track されており、ダウンロードではない)。
git clone https://github.com/sh202603/FlashVSR_plus
cd FlashVSR_plus

# 2. .venv の作成と依存の導入を 1 コマンドで行う。torch / torchvision は
#    pyproject.toml の設定で PyTorch の cu130 index から入る。
#    Python は開発ヘッダ(Python.h)を持つものが必須。FlashVSR の Triton
#    Sparse_SageAttention カーネルは実行時にヘッダを使って JIT され、ヘッダの無い
#    system / conda の Python では「fatal error: Python.h」で落ちる(さらに悪いと
#    tiny-long が黙って 0 フレームを返す)。fork は Python の版を固定していない
#    (requires-python >=3.10、.python-version なし)ので、uv-managed の standalone
#    Python を明示する。-dev パッケージ(python3.13-dev 等)を入れた system Python でもよい。
#    注意: uv 自体が snap 閉じ込めのアプリ(snap 版 VSCode 等)の中で動いていると、
#    managed Python は snap リビジョンのパス配下に置かれ、次の snap refresh で venv が
#    壊れる。その場合は安定パスのインタプリタを明示する:
uv sync --python 3.13 --python-preference only-managed     # または: uv sync --python /usr/bin/python3.13

# 3. 重み(~6.5 GB)は models/FlashVSR-v1.1/ に置かれる。初回実行時に HuggingFace から
#    自動ダウンロードされるので本手順は任意。jasna の処理中にダウンロードしたく
#    なければ先に取得しておく:
.venv/bin/huggingface-cli download JunhaoZhuang/FlashVSR-v1.1 --local-dir models/FlashVSR-v1.1

# 4. (推奨)jasna に組み込む前に FlashVSR 環境単体でスモークテスト。inline が使う
#    tiny-long / sage / bf16 を scale 2 で叩き、--accel で高速化の判定も確かめる。
#    手順3を省いた場合は重みのダウンロードも走る。run.py は出力先を作らないので先に作る:
mkdir -p _smoke
.venv/bin/python run.py -i ./inputs/example0.mp4 -s 2 -v 11 -m tiny-long -d cuda:0 -t bf16 -a sage --accel ./_smoke
```

補足:
- 手順4の起動ログに `[FlashVSR] accel: enabled fp8_conv_lq, fp8_dit, fused_dit.` が
  出れば高速化が使える。RTX 30 系以前の GPU では `... disabled: needs an FP8-capable GPU ...`
  と出て標準の処理で動く(異常ではない)。
- `sageattention` pip パッケージは**不要**。`-a sage` が使うのは fork が同梱する
  `sparse_sage` カーネルで、`sageattention` の import は guard 済み。
- 完了後 `<repo>/models/FlashVSR-v1.1/` に
  `diffusion_pytorch_model_streaming_dmd.safetensors`・`Wan2.1_VAE.pth`・
  `LQ_proj_in.ckpt`・`TCDecoder.ckpt`、隣に `<repo>/models/posi_prompt.pth` が揃う
  ——これが `--flashvsr-repo` の期待する構成。
- Windows では venv の Python が `.venv\Scripts\python.exe` になる(手順3・4の
  `.venv/bin/...` を読み替える)。そのほかの注意は「[Windows での注意事項](#windows-での注意事項)」。
- 既存の checkout を更新するときは `git pull` の後に `uv sync` をやり直す
  (高速化が使う `nvidia-cudnn-frontend` などの依存が追加されている)。

### 上流の checkout を使う場合

上流の [`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) を使う場合は、
手順1・2を次に読み替える(`uv.lock` が無いので版は固定されない):

```bash
git clone https://github.com/lihaoyun6/FlashVSR_plus
cd FlashVSR_plus
uv venv --python 3.13 --python-preference only-managed
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130   # CUDA 12.8 なら .../whl/cu128
```

この checkout では次の 2 点が fork と異なる。

- inline(`flashvsr-inline`)には tiny-long のパッチが要る(「[前提: tiny-long の修正](#前提-tiny-long-の修正)」)。
  オフライン(`flashvsr`)はパッチなしで動く。
- `--flashvsr-accel` は使えない。指定すると警告を出し、標準の速度で動く。

### jasna から指定するもの

- `--flashvsr-repo <path>`(必須): 上で作った `FlashVSR_plus` checkout。
- `--flashvsr-python <path>`(既定 `<repo>/.venv/bin/python`): 手順2の uv-managed
  venv の Python。
- `--flashvsr-model-dir <path>`(既定 `<repo>/models/FlashVSR-v1.1`): 重み。

## 使い方

```bash
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration flashvsr \
      --flashvsr-repo ~/FlashVSR_plus \
      --log-level info
```

### フラグ

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--flashvsr-repo` | (必須) | `FlashVSR_plus` checkout のパス。 |
| `--flashvsr-python` | `<repo>/.venv/bin/python` | FlashVSR 環境の Python(uv-managed standalone venv)。 |
| `--flashvsr-model-dir` | `<repo>/models/FlashVSR-v1.1` | FlashVSR 重みディレクトリ。 |
| `--flashvsr-version` | `11` | モデル版(`10` / `11`)。 |
| `--flashvsr-dtype` | `bf16` | 計算 dtype(`fp16` / `bf16`)。 |
| `--flashvsr-scale` | `4` | 両モード共通の処理倍率。`4` = モデルネイティブの 1024px、`2` = 512px(高速・低 VRAM)。詳細は「[処理倍率](#処理倍率--flashvsr-scale)」。 |
| `--flashvsr-accel` / `--no-flashvsr-accel` | off | 両モード共通: fork の高速化(FP8 と融合カーネル)を使う。RTX 40 系以降が必要で、それ以外は自動で標準の処理に戻る。詳細は「[高速化](#高速化--flashvsr-accel)」。 |
| `--flashvsr-lora` | なし | inline 専用: FlashVSR 用の Lada LoRA を使う(FlashVSR の過鮮鋭を抑える)。オフラインでは起動時にエラーになる。詳細は「[LoRA](#lora--flashvsr-lora)」。 |
| `--flashvsr-max-clip-frames` | `90` | オフライン専用: Phase 1 の `--max-clip-size` の上限(tiny モードの VRAM 対策)。inline は `--max-clip-size` をそのまま使う。 |
| `--flashvsr-unload-dit` / `--no-flashvsr-unload-dit` | on | VAE decode 前に DiT をオフロード(VRAM 節約)。 |
| `--flashvsr-tiled-vae` / `--no-flashvsr-tiled-vae` | on | FlashVSR の VAE decode をタイル化(VRAM 節約)。 |
| `--flashvsr-tiles` | `1` | inline 専用: DiT 推論を横短冊に分割して VRAM ピークを下げる(`2`〜`4`)。オフラインは無視する。詳細は「[strip タイリング](#strip-タイリング--flashvsr-tiles)」。 |
| `--flashvsr-bundle-dir` | temp | 中間 bundle をここに永続化(段階再開が可能に)。 |
| `--flashvsr-keep-bundle` | off | 完了後も bundle を残す(`--flashvsr-bundle-dir` 指定時は暗黙的に有効)。 |

色補正にフラグは無い(常時適用。後述)。

### 処理倍率(`--flashvsr-scale`)

FlashVSR は 4x モデルで、256px クロップを倍率ぶん bicubic で前拡大してから DiT が
そのサイズで復元する。`4` はモデルネイティブの 1024px 処理、`--flashvsr-scale 2` は
512px 処理になる。jasna のブレンドは復元フレームの実寸からクロップ幾何を導くので、
どちらの倍率でも他に変更なくフレームへ再合成され、出力動画の解像度は同じである。

scale 2 はオプトインのトレードオフである。モデルは 4x で学習されているため学習倍率
から外れるが、処理は大幅に軽い。この worker と逐語同一の lada-ex 実装での実測
(RTX 5080 16 GB、480p、`--tensorrt`)では、scale 2 + tiles 1 は scale 4 + tiles 2 に
対して e2e で**約 5 倍速**(壁時計 96 秒 vs 449 秒)、GPU 全体ピークは**約 4 GB 低い**
(11.3〜11.5 GB vs 14.8〜15.1 GB)。品質ゲートも同じものを通過した(flow-warping
error 比 1.128、ゲート ≤1.2。目視 A/B も合格)。既定は 4 のまま(初版の品質
ゲートを通したモデルネイティブの倍率)。

jasna 自身の実測(RTX 5080 16 GB / Linux、480p の実素材 4930 フレーム・全編
モザイク、clip 90 / overlap 8 の既定。全走行で出力フレーム数 = 入力。壁時計はコマンド
全体、VRAM は `nvidia-smi` の GPU 全体ピーク):

| モード | scale / tiles | 壁時計 | FlashVSR 時間 | VRAM ピーク | 備考 |
|---|---|---|---|---|---|
| 一次のみ | — | 20 秒 | — | 3.8 GB | 参照(fp8-recon) |
| inline | 4 / 2 | 491 秒 | 479 秒 | 13.7 GB | |
| inline | 2 / 1 | 113 秒 | 100 秒 | 10.1 GB | 4 / 2 比で **4.8 倍速、3.6 GB 低い** |
| inline、1080p(4203 f) | 2 / 1 | 156 秒 | 146 秒 | 11.3 GB | offload 0、アロケータ警告 0 |
| offline | 4 | 477 秒 | Phase 2 ~430 秒 | 13.2 GB | bundle 9.7 GB |
| offline | 2 | 194 秒 | Phase 2 ~150 秒 | 7.8 GB | |

以前のビルドは clip を 32 に上限化しており、FlashVSR 時間は上表の約 2 倍だった
(inline 4 / 2 で 841 秒、2 / 1 で 163 秒、offline 4 で 864 秒)。overlap 8 に対して
clip 32 は 1 clip で 16 フレームしか前進せず、DiT フレーム数が 1.8 倍、呼び出し回数が
3 倍になるためで、worker の差ではない(移植前 worker との A/B は一致。色補正の
コストは +6〜7%)。上限撤廃による VRAM 増は一次側のキュー分のみ(480p +0.1〜0.3 GB、
1080p +0.7 GB)。

### 高速化(`--flashvsr-accel`)

`--flashvsr-accel` を付けると、fork の高速化(fork の `--accel` と同じもの)を両モードで使う。
fork の計測(RTX 5060 Ti 16 GB、tiny-long、90 フレーム)では、FlashVSR の処理が
scale 2 で 1.37 倍、scale 4 で 1.41 倍速くなり、FlashVSR 単体のピーク確保量が約 1.4 GiB 減った。
既定は無効。

置き換えるのは次の 3 つで、それぞれ起動時に使えるかを判定する:

| 部品 | 内容 |
|---|---|
| `fp8_conv_lq` | LQ projector の畳み込みを FP8 で計算(cuDNN graph API) |
| `fp8_dit` | DiT の Linear と FFN を FP8 で計算 |
| `fused_dit` | DiT の RMSNorm + RoPE と AdaLN を Triton の融合カーネルで計算 |

VAE デコーダ(TCDecoder)の FP8 化は含まない。TCDecoder は出力の画素を直接作る段で、
FP8 の粗い仮数が肌などのなめらかな階調を段にし、縞として見えるためである。

**要件**: RTX 40 系以降の GPU(sm89 以上)、`--flashvsr-version 11` と
`--flashvsr-dtype bf16`(どちらも既定)、fork の checkout。FP8 畳み込みは cuDNN 9.17 以上を
要するが、fork の torch(cu130)に同梱の cuDNN で満たす。

**自動フォールバック**: 要件を満たさない部品は、起動時の判定(GPU、dtype、ライブラリ、
試しのビルド、warmup)で外れ、標準の処理で動く。全部品が外れた実行の出力は、
`--flashvsr-accel` なしの実行とビット単位で一致する。実行中に部品が失敗した場合は、
その部品を以後標準の処理に戻し、その clip を 1 回やり直す(inline は worker、
オフラインは Phase 2 driver がやり直す)。VRAM 不足(OOM)では部品を外さない。

**確認のしかた**: inline の worker は標準出力を捨てるので、起動時の判定結果を jasna に返し、
jasna がログに出す。`--log-level info` で次のように表示される:

```text
[flashvsr-inline] [FlashVSR] accel: enabled fp8_conv_lq, fp8_dit, fused_dit.
[flashvsr-inline] worker ready (acceleration: fp8_conv_lq, fp8_dit, fused_dit)
```

外れた部品は、理由付きの警告(`... disabled: <理由>; using the standard path.`)になる。
実行中に外れた部品も警告になる。どちらも `--log-level warning` 以上の詳しさで表示される
(既定の `error` では出ない)。オフラインは Phase 2 の出力に fork のログがそのまま出る。

**出力**: 標準の処理と少し異なる(FP8 の丸めによる微差を、sparse attention が増幅する)。
fork の検証では、flow-warping error の比が標準の 1.07 倍以内(品質ゲートは 1.2 以下)で、
目視 A/B でも標準と同等だった。

**jasna での実測**: Windows 11 / RTX 5060 Ti 16 GB、1080p の実素材(6242 フレーム)、
inline scale 2 / tiles 1。2 回を続けて計測し、GPU 全体のピークはデスクトップ常駐(約 3.2 GB)込みの値:

| | 壁時計 | GPU 全体ピーク |
|---|---|---|
| `--flashvsr-accel` なし | 577 s | 12191 MiB |
| `--flashvsr-accel` あり | **463 s(1.25 倍速)** | **11311 MiB(−880 MiB)** |

どちらも出力は 6242 フレームで、offload と worker のリトライは 0 回だった。
2 本を並べて再生した目視 A/B でも、画質の差は見られなかった。

scale 4 / tiles 1(同じ 1080p 素材、高速化あり)も測った。完走し(1555 s、出力 6242 フレーム、
OOM・リトライ・offload はいずれも 0)、tiles 2 の高速化なし(2519 s)より 1.62 倍速かった。
ただし GPU 全体のピークは 15874 MiB で、天井まで 437 MiB しかない。しかもこれはデスクトップ常駐が
1.36 GB と少ない状態での値で、常駐が 2 GB を超えると足りなくなる。そのため、16 GB カードで scale 4 を
使うときは、高速化ありでも `--flashvsr-tiles 2` を推奨する。
壁時計の短縮は fork の 1 clip 単位の計測(1.37 倍)より小さい。一因は、一次のパイプラインと
デコード・エンコードの時間が変わらないことである(高速化なしで FlashVSR 時間は壁時計の約 9 割)。
ただしこれだけでは 1.32 倍程度までしか下がらないので、GPU を一次と分け合うことの影響もあると考えられる。

同じ構成を Linux でも測った: Ubuntu 26.04 / RTX 5080 16 GB / driver 595.91.07、
1080p の実素材(4203 フレーム)、inline scale 2 / tiles 1。
GPU 全体のピークはデスクトップ常駐(約 1.8 GB)込みの値:

| | 壁時計 | GPU 全体ピーク |
|---|---|---|
| `--flashvsr-accel` なし | 162 s | 11581 MiB |
| `--flashvsr-accel` あり | **127 s(1.28 倍速)** | **10138 MiB(−1443 MiB)** |

どちらも出力は 4203 フレームで、offload・worker のリトライ・実行中の部品の脱落はいずれも 0 回
だった。常駐の差(114 MiB)を除いたアプリ分のピークは 9695 → 8366 MiB で、fork の公称どおり
約 1.4 GiB 減る。2 本を並べて再生した目視 A/B でも、画質の差は見られなかった。

Linux の scale 4 は、高速化ありで tiles 2 と tiles 1 の両方を測った(同じ 1080p 素材)。
tiles 2 は 556 s / 13765 MiB で offload・OOM 警告ともに 0 件、tiles 1 は 443 s と速いものの、
ピークが 15770 MiB の天井に張り付き、offload 63 回(1426 MiB)と OOM 警告 177 件を伴った
(高速化なしの tiles 1 は 159 回・995 件)。高速化は消費を下げるが天井は越えられないので、
Linux でも scale 4 は `--flashvsr-tiles 2` を推奨する。

オフライン 3 段にも効果が出る。480p の実素材(4931 フレーム)の scale 2 で、高速化なしの
194 s / 7770 MiB に対し、高速化ありは 139 s(1.40 倍速)/ 7440 MiB だった。Phase 2 の出力に
3 部品が有効になったログが出て、3 段とも完了し、出力フレーム数も入力と一致した。

補足:
- 部品ごとの環境変数(`FLASHVSR_FP8_CONV` / `FLASHVSR_FP8_DIT` / `FLASHVSR_FUSED_DIT`)は
  worker にそのまま渡る。A/B 検証で 1 部品だけ外すときは `FLASHVSR_FP8_DIT=0 jasna ...` の
  ようにコマンド単位で渡す。`--no-flashvsr-accel`(既定)は `FLASHVSR_ACCEL` を取り除くので、
  シェルに `FLASHVSR_ACCEL=1` が残っていても高速化は有効にならない。
- オフラインで bundle から再開する場合、完了済みの clip はそのまま使われる。高速化の設定を
  変えて再開すると、clip ごとに高速化の有無が混ざる(clip は互いに独立なので継ぎ目は
  出ない)。揃えたい場合は新しい bundle で実行し直す。
- 上流の checkout で指定すると、高速化が無い旨の警告を出して標準の速度で動く。

### LoRA(`--flashvsr-lora`)

`--flashvsr-lora` は、FlashVSR の DiT に Lada の LoRA を適用する。
公開している `lada_flashvsr_secondary_lora_v1.pt`(30 MB)は、DiT の attention と FFN の
Linear に掛ける rank 16 の LoRA で、LQ projector と TCDecoder は変えない。
拡大したクロップを元の大きさに合成したとき、原寸の実際の肌の肌理と統計がそろうように学習し、
素の FlashVSR との目視比較で選んだ。既定は無効。

jasna の `model_weights` ディレクトリに置き、ファイル名で指定する:

```bash
wget -O model_weights/lada_flashvsr_secondary_lora_v1.pt \
  https://huggingface.co/sh202603/lada-seedvr2-lora/resolve/main/lada_flashvsr_secondary_lora_v1.pt

jasna --input in.mp4 --output out.mkv --secondary-restoration flashvsr-inline \
      --flashvsr-repo ~/FlashVSR_plus --flashvsr-scale 2 \
      --flashvsr-lora lada_flashvsr_secondary_lora_v1.pt
```

パスを含まないファイル名は `model_weights` ディレクトリから探す。パスでの指定もできる。

**変わること**(scale 2、一次は BasicVSR++。素の FlashVSR に対して、合成後のフレームで計測):

- 素の FlashVSR の中帯域の過鮮鋭(「シャープナー」のような見え方)が、4〜8 px の帯で約 3 割減る。
  粒は密になり、尖りが減る。
- クロップの拡大が小さい領域(フレームへ約 1.5 倍で戻す領域)では、素の FlashVSR は周囲の実際の肌より
  多くの粒を足す。LoRA はそれを周囲と同じ水準に戻す。
- 時間方向のちらつきが少し減る(約 5%)。
- 代わりに、最も細かい粒の一部を失う(約 3 倍で戻す領域の 1〜2 px 帯で約 35%)。
  素の FlashVSR が素材によって鮮鋭すぎると感じるときに使い、細部を最大限に残したいときは使わない。
- 残存モザイクの出方は変わらない。

**コスト**: LoRA は worker の起動時に bf16 の低ランクアダプタとして適用する(約 5% 遅くなり、
VRAM が 32 MB 増える)。モデルの重みには合成しない。学習した変化量が bf16 や FP8 の重みの
分解能よりはるかに小さく、合成すると大半が丸めで消えるためである。

**`--flashvsr-accel` との併用**: できる。アダプタは FP8 の Linear の横に置き、融合 FP8 FFN には
差し込む形で適用する。効き方は高速化なしと同じだった(実測)。実行中に FP8 の DiT 部品(`fp8_dit`)が
失敗すると、fork のフォールバックが標準の Linear を戻し、アダプタも一緒に外れてしまう。
その場合 worker は処理を止め、jasna が `fp8_dit` を切った worker を起こし直して(LoRA は再適用される)、
その clip を 1 回やり直す。このとき警告が出る。

**対象**: inline モードのみ。`--secondary-restoration flashvsr` で `--flashvsr-lora` を指定すると
起動時にエラーになる。この LoRA は一次が BasicVSR++ の場合に合わせて選んだもので、オフラインの
主な用途である SeedVR2 一次では、FlashVSR が足す細かい粒の大半を消してしまうため提供しない。
`--flashvsr-scale 2` で検証済み(scale 4 は未検証)。

worker は起動時に LoRA の適用をログに出す:

```text
FlashVSR worker: applied LoRA lada_flashvsr_secondary_lora_v1.pt (rank 16, step 1000) to 180 DiT linear layers
```

**jasna での確認(Linux)**: Ubuntu 26.04 / RTX 5080 16 GB / driver 595.91.07、1080p の
実素材(6242 フレーム)を inline scale 2・`--flashvsr-accel` ありで、`--flashvsr-lora` の
有無で処理した。どちらも 6242 フレームを同じエンコード設定(HEVC Main 10)で出力し、
LoRA ありの出力は約 7% 小さかった(3.61 Mbps と 3.89 Mbps)。並べて再生すると、LoRA ありでは
毛の過鮮鋭な見え方が抑えられ、上に書いた効果と一致した。

### 色補正

FlashVSR が生成したクロップは、元になった一次復元結果から色味がずれることがあり、
ブレンド後に復元領域と周囲の色調差として見える。そのため両モードとも、各出力
クロップを常に**入力クロップ(一次出力)の bicubic 拡大**を参照に補正する。方式は
wavelet 再構成で、FlashVSR 出力の高周波(テクスチャ)を入力の低周波(局所的な
色調)の上に載せる。適用は clip ごとにクロップ全体へ 1 回(短冊ごとには行わない)、
量子化の前で、両モードが同じ関数を使う。

実測は、二次が変更した画素内のチャネル毎 median |Δmean|(8bit、同じ clip 構成の
一次のみ出力に対して)。lada-ex: 補正なし 5.28 → AdaIN 0.98 → **wavelet 0.34**
(scale 2 では 0.32)。jasna の scale 2(480p の実素材): inline 1.76 → 0.43 →
**0.24**、offline 1.61 → **0.24**(関数を共有する両モードが同じ値に落ちる)。
上流の FlashVSR_plus にも `color_fix` はあるが jasna は使わない。呼び出しが裸の
`except: pass` で包まれており、失敗しても「未適用」と区別が付かないためである。

Windows(RTX 5060 Ti、1080p、inline scale 2 / tiles 1)でも効果と代償を確認した。
指標は上記と別実装(復元領域のマスクを none 出力基準で決め、none と wavelet を同一
フレーム・同一マスクで測る対応比較)なので絶対値は上表と比較できないが、
none 5.88 → **wavelet 3.20**(比 0.54)で、135 フレーム中 **91.1%** で wavelet が
none を下回った。コストは FlashVSR 時間の **+5.3%**(wavelet 524.5 s 対 none 498.3 s)で、
Linux の +6〜7% と整合する。

CLI フラグは無い。A/B 検証専用に、環境変数
`JASNA_FLASHVSR_COLOR_FIX=adain|wavelet|none` で両モードの方式を上書きできる。
シェルに設定したまま戻し忘れると以後の走行が全て上書き値で回るので、
`JASNA_FLASHVSR_COLOR_FIX=none jasna ...` のようにコマンド単位で渡すこと。

### clip 長

clip の長さは通常どおり `--max-clip-size`(既定 90)で決まる。

- **inline**: 上限なし。worker の tiny-long は VRAM が clip 長に対して平坦で、
  一次側のコストは FlashVSR なしの実行と同じ([tuning](tuning.md))。
- **offline**: Phase 2 の tiny モードは原理上 clip 長に比例するため
  `--flashvsr-max-clip-frames`(既定 90)で上限化する。実測では 32 → 90 で Phase 2 の
  ピークは平坦(scale 4 で 13.1 → 13.2 GB、scale 2 で 7.8 GB のまま)。90 超は未測定。
  Phase 2 が OOM する環境ではこの値を下げる。

## ディスク容量

bundle の容量は Phase 2 の**非圧縮の拡大出力**が支配する。復元クロップ 1 枚が既定の
scale 4 で 1024×1024×3 ≈ **3 MiB**(`--flashvsr-scale 2` ではその 1/4 の 0.75 MiB)
で、256px 一次 dump は 1 clip 丸ごとで ~3 MiB。つまり bundle 容量はモザイク・クロップ
枚数に比例し、動画が長いほど増える(以下の数値は scale 4):

- 目安: **モザイクを含む 1 ソースフレームあたり ~4 MB** ≒ 全編モザイクの 30fps 動画で
  **1 分あたり ~8 GB**(モザイクが一部の時間帯だけなら比例して少ない)。
- 実測: 6 分 / 10,661 フレーム・全編モザイク・510 clip → **~46 GB**
  (1024px 出力 ~45 GB + 256px dump ~1.6 GB)。
- 長尺・モザイク多めの動画では **数百 GB** に達しうる。

ピークは **bundle 全量**。Phase 2 が全 clip の 1024px を書き終えてから Phase 3 が
始まるため、全 fvsr が同時にディスク上に存在する。

> ⚠️ **既定の bundle はシステム temp(`/tmp`)下に作られる。Linux では `/tmp` が
> しばしば `tmpfs`(RAM 上)で数十 GB しかない。**大きな bundle をそこに書くと
> `/tmp` が溢れ RAM を食い潰して失敗する。短いクリップ以外では
> `--flashvsr-bundle-dir <path>` で bundle 全量が入る実ディスクを指すこと
> ——目安は *モザイク分数 × 8 GB*。段階再開も可能になる。

jasna は自動でこれを見張る: Phase 1 前に bundle dir が tmpfs なら警告し、空き容量と
最悪ケース見積りを表示。さらに Phase 1 後(実 clip 数が判明後)に選択した scale での
出力の正確なサイズを計算し、**入り切らなければ高コストな Phase 2 を始める前に中断する**(bundle は
保持されるので `--flashvsr-bundle-dir` を大きいディスクに向けて再開できる)。

## 制約

- **ファイル出力専用**。`--stream`・フォルダ/画像入力・`--frame-gen` とは併用不可
  (フレーム生成は出力に対する別パスとして実行する)。
- **fps リターゲット・スマートレンダリング・VR 非対応**。`--retarget-high-fps`
  (Phase 1 の frame stride が Phase 3 の再ブレンド索引とずれる)・`--segments`・
  VR 処理(`--vr-mode sbs`/`sbs-fisheye`、または `auto` が VR コンテンツを検出した
  場合 — Phase 3 に VR プロジェクタがない)は起動時に拒否する。
- **encode が 2 回**。Phase 1 は完全にテスト済みのパイプラインをそのまま流すため
  捨て出力を encode し、最終 encode は Phase 3 で行う。通常実行より encode が 1 回多い。
- **同梱なし / サポーターモデルとは無関係**。FlashVSR は独自ライセンスのサードパーティ
  モデル。checkout・重み・venv は利用者が用意する。jasna のサポーターモデルとは無関係。

## inline モード(`--secondary-restoration flashvsr-inline`)

オフライン 3 段と同じ FlashVSR checkout / 重み / venv・同じ `--flashvsr-*` フラグ
(`repo` / `python` / `model-dir` / `version` / `dtype` / `scale`)を使うが、**中間
ファイルを一切作らず**、jasna の通常のストリーミングパイプラインの中で FlashVSR を
二次復元として走らせる。`basicvsrpp` 一次 + 16 GB カード向けのモードで、SeedVR2 一次
とは併用できない(常駐 worker 2 つで 16 GB を超える。その構成はオフラインモードで
組む)。

```bash
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration flashvsr-inline \
      --flashvsr-repo ~/FlashVSR_plus \
      --log-level info
```

### オフラインとの違い

| | `flashvsr`(オフライン 3 段) | `flashvsr-inline` |
|---|---|---|
| パス | dump → FlashVSR → reblend の 3 プロセス | 単一ストリーミングパス |
| 中間ファイル | 256px + 拡大クロップの bundle(数十 GB 級) | **無し** |
| encode 回数 | 2(捨て + 最終) | 1 |
| FlashVSR モード | tiny(O(T)、~12–16 GB) | **tiny-long(O(1)、~11.9 GB)** |
| 必要 VRAM | 各段が非同時なので実質 tiny 単体分 | primary と**同時常駐**(scale 4: tiles 2 で ~13.7 GB、tiles 無しは 16 GB の天井。scale 2: ~10〜11 GB) |
| FlashVSR checkout | パッチ不要 | 推奨 fork はそのまま使える。上流の checkout は **tiny-long マルチチャンク修正のパッチ必須** |
| 段階再開 | 可(bundle 永続化) | 不可(単一パス) |
| 進捗 / キャンセル / GUI | 3 段フロー | 通常 secondary と同じ |

### 前提: tiny-long の修正

inline は VRAM 定常(O(1))の **tiny-long** を使う。上流の FlashVSR_plus の tiny-long は
第 2 チャンクで壊れる既知バグ(`8192 vs 4096` エラー)があり、**修正を含む checkout が
必須**。推奨の fork([`sh202603/FlashVSR_plus`](https://github.com/sh202603/FlashVSR_plus))は
修正を含むので、何もしなくてよい。jasna は起動時に checkout を検査し、修正が無ければ
明示エラーで停止して、fork・パッチ・`flashvsr`(オフライン、tiny、パッチ不要)を案内する。

上流の checkout を使う場合は、同梱のパッチ
[`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../../patches/flashvsr_plus_tinylong_multichunk_fix.patch)
を当てる:

```bash
cd ~/FlashVSR_plus
git apply /path/to/jasna/patches/flashvsr_plus_tinylong_multichunk_fix.patch
```

やっていることは 2 箇所のチャンク跨ぎキャッシュ clear を無効化するだけ
(`src/pipelines/flashvsr_tiny_long.py` の per-chunk `LQ_proj_in.clear_cache()` と
`TCDecoder.clean_mem()` を削除。ループ前の一度きりのリセットは残す)。

### 挙動と制約

- **frame-gen off を強制**(オフラインと同じ理由)。
- **fp8-recon を自動有効化**(未指定時)。一次のピークを ~0.9–1.7 GB 下げ、同時常駐の
  予算に収める。GPU が fp8 非対応(sm89 未満 / `--fp16` 無し)なら TRT へフォールバック。
- 同期実行。FlashVSR(~15 crop-fps)が律速なので、モザイクが多い区間はその速度に
  律速される(モザイクの無いフレームは一次のみで高速)。FlashVSR が壁時計を支配する
  ため、`--batch-size` を下げても速度低下はほぼ無い。
- VRAM(**16 GB カード + デスクトップ常駐**時、既定の scale 4): **tiles 無しは
  16 GB では使わない**。480p でも 1080p でも物理天井に張り付き(実測 ~15.8 GB)、
  `expandable_segments: memory mapping failed with OOM` の警告と offload が出た上で、
  clip の途中で worker が実際に OOM することがある(1080p / clip 90 で 57 clip 中
  32 回)。worker はその clip を 1 回リトライし、再失敗なら停止する(以前は失敗した
  clip の残りを最後のフレームの複製で埋めていたため、残像として見えていた)。
  **`--flashvsr-tiles 2`** を使う(次節。480p 13.7 GB、1080p 14.7 GB、offload 0)。
  VRAM が少ない環境や、パッチを当てていない上流の checkout ではオフライン(`flashvsr`)を使う。**`--flashvsr-scale 2` では様相が変わる**: 分割なしで
  Linux の 480p 10.1 GB、1080p 11.3 GB(clip 90。offload 0、アロケータ警告 0)で、
  タイリングも天井対策も要らない。全表は「[処理倍率](#処理倍率--flashvsr-scale)」。
- **Windows で `expandable_segments` が使えない影響は scale 4 でのみ出る**。scale 4 は
  worker の reserved が ~13 GB に膨らんで tiles 無しの inline が物理天井に張り付くため、
  1080p では **`--flashvsr-tiles 2` を推奨**。scale 2 は断片化の影響を受けず、Windows の
  実測が Linux を下回る(1080p 10.8 GB、天井まで 5.5 GB)ので **tiles 1 のままでよい**。
  実測は「[Windows での注意事項](#windows-での注意事項)」。

### strip タイリング(`--flashvsr-tiles`)

inline 専用の VRAM 対策。各 256px クロップを幅そのままに高さ方向だけ横短冊(strip)に
分割し、短冊ごとに tiny-long を回して羽根(feather)合成する。DiT のトークン活性メモリ
(特に block-sparse draft の attn マスク)はタイル面積の二乗で減るため、少ない計算増で
ピーク VRAM が下がる。オフライン(`flashvsr`)は本フラグを無視する。

| `--flashvsr-tiles` | 短冊 | attn マスク(対 full) | 計算量(対 full) |
|---|---|---|---|
| `1`(既定) | なし(単発) | 1.0 | 1.0 |
| `2` | 2 枚(各 256w×160h) | 0.39x | ~1.25x |
| `3` | 3 枚(各 256w×128h) | 0.25x | ~1.5x |
| `4` | 4 枚(各 256w×96h) | 0.14x | ~1.5x |

短冊は少ないほど速く品質も良い(重複計算が少なく、1 短冊の空間文脈が広い)ので、
VRAM が許す**最小の枚数**を選ぶ。2 で収まれば 2、天井に張り付く/OOM するなら 3、
それでも足りなければ 4。

上表は scale 4 のもので、短冊高は 32 の倍数に丸められる(4 倍に拡大した短冊が DiT の
要求する 128 の倍数になる)。`--flashvsr-scale 2` では丸め粒度が 64 になるため短冊は
やや大きめに丸まり(tiles `2` = 256w×192h × 2 枚、重複 128px。`3` = 128h × 3 枚、
`4` = 128h × 4 枚)、重複計算の比率が上がる。ただし scale 2 は分割なしのピークが
天井から十分離れるため、タイリングが要る場面はまず無い。

品質への影響: 短冊境界は羽根合成され、実機確認(Windows / RTX 5080、tiles 1 との
同一フレーム比較)ではバンディング・段差・短冊間の色調ずれは検出されなかった。差分は
拡散モデルの確率的なテクスチャ揺らぎの範囲にとどまる。なお合成の都合上、出力の最外
1px は重み 0 になる(本家 run.py 由来の既存挙動。ブレンド時のクロップ境界は羽根が
かかるため実害は小さい)。

### 実装

- 同期 `SecondaryRestorer`: `jasna/restorer/flashvsr_inline_secondary_restorer.py`
  (FlashVSR venv worker を resident spawn、length-prefixed の uint8 BGR wire で
  RGB 反転はこちら側、`close()` で終了)。
- worker(FlashVSR venv 実行、jasna 非依存): `jasna/restorer/flashvsr_inline_worker.py`
  (tiny-long pipe、`imageio.get_writer` を差し替えてロスレスにテンソル捕獲、
  small clip は next_8n5 パディングで吸収し厳密に T 枚返す。strip の分割と
  羽根合成、色補正、高速化の判定結果の中継もここ)。このファイルは lada-ex の `flashvsr_worker.py` と
  **バイト単位で同一に保つ**(SeedVR2 worker と同じ方針)。wire が lada ネイティブの
  BGR なのはそのためで、FlashVSR_plus 側の互換破壊は片方で直して diff コピーする。
- CLI 配線: `jasna/main.py`。テスト: `tests/test_flashvsr_inline.py`。

## Windows での注意事項

検証環境は 2 つ。scale 4 / tiles の初回検証が **Windows 11 / RTX 5080 16 GB**、
scale 2 と色補正の検証が **Windows 11 / RTX 5060 Ti 16 GB / driver 616.92**
(どちらも torch 2.13.0+cu130 の FlashVSR venv)。GPU が異なるため**壁時計は
両者間で比較できない**が、VRAM ピークは確保サイズが GPU 型番に依らないため比較できる。

結論: **16 GB カードの 1080p は `--flashvsr-scale 2 --flashvsr-tiles 1` が第一選択**
(ピーク 10.8 GB、天井まで 5.5 GB)。scale 4 を使う場合はオフライン(`flashvsr`)か
inline + `--flashvsr-tiles 2` にする。scale 4 の tiles 無し inline は完走はするが
余裕がほぼ無い。

- **PyTorch の `expandable_segments` は Windows 未対応**(警告を出して既定の
  キャッシングアロケータへフォールバック)。`backend:cudaMallocAsync` でも改善しない
  (実測でむしろ微増)。jasna は Windows では worker に `expandable_segments` を
  設定しない。
- **断片化のペナルティは scale 依存**で、scale 4 でのみ現れる。デスクトップ常駐を
  差し引いたアプリ分で Linux と比べると:

  | 構成 | Linux | Windows | 差 |
  |---|---|---|---|
  | inline scale 2 / tiles 1(1080p) | 9642 MiB | 8925 MiB | **-717** |
  | offline Phase 2 scale 2(480p) | 6070 MiB | 6418 MiB | +348 |
  | offline Phase 2 scale 4(480p) | 11494 MiB | 13040 MiB | **+1546** |

  scale 4 は 1024px で処理するため断片化が効き、tiny-long の reserved が Linux の
  「フラット ~11.9 GB」より **+1〜2 GB** 膨らむ。scale 2 は 512px 処理でこれが起きず、
  Windows のほうがむしろ低い。**「Windows は一律 +1〜2 GB」という見積りは
  scale 2 には当てはまらない。**
- **WDDM デスクトップ常駐が ~1 GB** を取る(ヘッドレス Linux ではほぼ 0)。16 GB
  カードの実効空きは **~15.2 GB**。ブラウザや IDE も開いた実デスクトップでは
  アイドルで ~2 GB を超えることもある。
- inline のフルパイプライン実測(フル長素材、nvidia-smi の GPU 全体ピーク、
  アイドル ~2.1 GB の実デスクトップ常駐):

  | `--flashvsr-tiles` | 1080p ピーク | 壁時計(対 tiles 1) |
  |---|---|---|
  | `1` | 15918 MiB | 1.00x |
  | `2` | **14222 MiB** | 1.25x |
  | `3` | 12490 MiB | 1.46x |
  | `4` | 11514 MiB | 1.46x |

  tiles `1` は 480p でも 1080p でも完走した(offload 0 回、OOM 警告 0 件)が、
  ピークは物理天井(16303 MiB)まで 400 MiB を切り、常駐アプリの変動で OOM に
  転じうる。**scale 4 での 1080p 常用は `--flashvsr-tiles 2`**(余裕 ~2 GB、減速 +25%)。
  短冊境界のシーム(バンディング、色調ずれ)はこの実測でも検出されなかった。
- scale 2 と scale 4 の実測(RTX 5060 Ti 16 GB、常駐 1870 MiB、GPU 全体ピーク。
  480p = 852x480 / 10661 フレーム、1080p = 1920x1080 / 6242 フレーム):

  | モード | scale / tiles | 解像度 | ピーク | 天井までの余裕 | 壁時計 |
  |---|---|---|---|---|---|
  | inline | 2 / 1 | 480p | 10237 MiB | 6074 MiB | 824 s |
  | inline | 2 / 1 | 1080p | **10795 MiB** | **5516 MiB** | 575 s |
  | inline | 4 / 2 | 1080p | 15148 MiB | 1163 MiB | 2519 s |
  | offline | 2 | 480p | 8288 MiB | 8023 MiB | 1220 s |
  | offline | 4 | 480p | 14910 MiB | 1401 MiB | 3174 s |

  全走行で offload 0 回、worker の clip リトライ 0 件、出力フレーム数は入力と一致。
  scale 4 / tiles 2 の壁時計は scale 2 / tiles 1 の 4.38 倍で、Linux の 4.35 倍と一致する。
  **scale 4 側(inline 15148 MiB、offline 14910 MiB)は常駐 1.9 GB 込みの値**なので、
  常駐が 2.5 GB あるマシンでは天井まで 500 MiB を切る。scale 4 を 16 GB で回すなら
  常駐アプリの管理が前提になる。
- オフラインの bundle 実サイズは見積りどおり(480p / 10661 フレーム):
  scale 2 は見積り 7.0 GiB に対し実測 8.1 GiB、scale 4 は見積り 28.1 GiB に対し
  実測 30 GiB(1024px 出力 29 GiB + 256px dump 1000 MB)。どちらも
  「見積り + dump 分」の範囲に収まる。Phase 2 の `UnicodeEncodeError` は発生しない。
- 実測ピーク(scale 4 / tiny-long / bf16 / sage / 85 フレーム、reserved 値):
  - **256px 入力(jasna の実ワークロード): ~13.0 GB** — Phase 2 は GPU を単独占有
    するので、オフラインは 16 GB Windows で動く。
  - **384px 入力(同梱 example0 での smoke): ~15.1 GB** — 空きと紙一重。ブラウザや
    IDE が数百 MB 使っているだけで OOM する。**smoke の OOM ≠ jasna 実負荷の OOM**。
    セットアップ手順 4 の smoke が scale 2 なのはこのため。
- **`-m tiny`(O(T))での 85 フレーム smoke は 16 GB Windows では OOM して正常**。
  smoke は手順 4 のとおり `-m tiny-long` で行う。
- venv の Python は `<repo>/.venv/Scripts/python.exe`(`--flashvsr-python` の既定も
  Windows ではこのパスに解決される)。
- stdout がパイプに向く(リダイレクト / 一部の GUI 起動)と、FlashVSR の起動バナー
  (ブロック文字)が cp932 で `UnicodeEncodeError` になり推論前に落ちる。jasna からの
  起動(オフライン Phase 2 / inline worker)は `PYTHONUTF8=1` を自動設定するので
  対処不要。**run.py を手で叩いて出力をリダイレクトする場合は
  `$env:PYTHONUTF8=1` を先に設定**する。
- run.py は出力ディレクトリを自動作成しない(存在しないと推論後の書き出しで
  `FileNotFoundError`)。事前に `mkdir` しておく。

## 実装(オフライン)

- オーケストレータ・bundle 形式・Phase 1 dump hook・Phase 3 reblend:
  `jasna/restorer/flashvsr_offline.py`。
- Phase 2 driver(FlashVSR venv): `jasna/restorer/flashvsr_phase2_driver.py`。
- サブプロセス分岐: `jasna/__main__.py`(`--flashvsr-phase`)。
- CLI 配線 / 早期分岐: `jasna/main.py`。
- テスト: `tests/test_flashvsr_offline.py`、`tests/test_main.py`。

再利用した jasna 資産: `BlendBuffer` / `crop_buffer.scale_offsets`(拡大クロップは
どちらの scale でも無改変で再 blend)、`RestorationPipeline.build_secondary_result`
(`[keep_start:keep_end]` スライス)、`pipeline_items`(直列化単位)、Phase 3 の
decode/encode に `media/backend.make_video_{reader,encoder}`。
