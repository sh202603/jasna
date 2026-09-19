# FlashVSR 二次復元(`+modi`)

`--secondary-restoration flashvsr` / `flashvsr-inline` は、一次復元された 256px の
モザイククロップを [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
(one-step streaming diffusion VSR。jasna は
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) fork を使用)
で拡大し、一次の BasicVSR++ では大きなモザイク領域・接写・4K 素材でぼやけがちな
テクスチャの写実性を補う。処理解像度はモデルネイティブの 1024px(4x)か、
`--flashvsr-scale 2` で 512px。どちらでもブレンドがクロップを元の領域へ縮小合成する
ため、出力動画の解像度は変わらない。出力クロップは常に、元になった一次復元結果を
参照して色補正される(「[色補正](#色補正)」)。

FlashVSR には 2 つのモードがあり、どちらもサポートされる。構成で選ぶ:

| | `flashvsr-inline`(単一パス) | `flashvsr`(オフライン 3 段) |
|---|---|---|
| 向く構成 | `basicvsrpp` 一次 + 16 GB カード。単一パスで**中間ファイル・ディスクゲート・二重 encode が無い** | SeedVR2 一次との併用(最高品質構成)、12 GB 級 GPU、段階再開が要る長尺 |
| FlashVSR パイプライン | tiny-long(VRAM がクリップ長に依存しない。**tiny-long パッチ必須**) | tiny(パッチ不要) |
| `--restoration-model-name seedvr2` との併用 | 起動時エラー(常駐 worker 2 つで 16 GB 超過) | 可 |

以下は主にオフラインモードの説明で、inline は末尾の「inline モード」で扱う。

オフライン 3 段が存在する理由: FlashVSR の tiny モードは**単体で 12–16 GB VRAM** を
消費するため、jasna の一次パイプラインと 16 GB カード上で同時常駐できない。ピーク
VRAM が時間的に重ならないよう処理をプロセス分割することで初めて収まる。inline モードは
FlashVSR の **tiny-long**(定メモリ ~11.9 GB、パッチ要)を使い、一次(fp8-recon で
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
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) fork の
checkout・重み・専用仮想環境を利用者が用意し、`--flashvsr-repo` で jasna に渡す。

### FlashVSR checkout のセットアップ(一度だけ)

RTX 5080(sm120, 16 GB)/ Linux / CUDA 13.0 で検証済みの再現手順。torch
2.13.0+cu130 / triton 3.7.1 になる:

```bash
# 1. jasna が対象とする fork を clone。models/posi_prompt.pth もこれで入る
#    (repo に git-track されており、ダウンロードではない)。
git clone https://github.com/lihaoyun6/FlashVSR_plus
cd FlashVSR_plus

# 2. Python 開発ヘッダ(Python.h)を持つ Python で venv を作る。これは必須。FlashVSR の
#    Triton Sparse_SageAttention カーネルは実行時にヘッダを使って JIT され、ヘッダの
#    無い system / conda の Python では「fatal error: Python.h」で落ちる(さらに悪いと
#    tiny-long が黙って 0 フレームを返す)。uv-managed の standalone Python か、-dev
#    パッケージ(python3.13-dev 等)を入れた system Python のどちらかを使う。
#    注意: uv 自体が snap 閉じ込めのアプリ(snap 版 VSCode 等)の中で動いていると、
#    managed Python は snap リビジョンのパス配下に置かれ、次の snap refresh で venv が
#    壊れる。その場合は安定パスのインタプリタを明示する:
uv venv --python 3.13 --python-preference only-managed     # または: uv venv --python /usr/bin/python3.13

# 3. CUDA に合う wheel index で FlashVSR の依存を venv に入れる
#    (jasna は cu130 で検証。CUDA 12.8 なら .../whl/cu128)。
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130

# 4. 重み(~6.5 GB)は models/FlashVSR-v1.1/ に置かれる。初回実行時に HuggingFace から
#    自動ダウンロードされるので本手順は任意。jasna の Phase 2 中にダウンロードしたく
#    なければ先に取得しておく:
.venv/bin/huggingface-cli download JunhaoZhuang/FlashVSR-v1.1 --local-dir models/FlashVSR-v1.1

# 5. (推奨)jasna に組み込む前に FlashVSR 環境単体でスモークテスト。jasna の Phase 2
#    が使う tiny / sage / bf16 の 4x パスそのものを叩き、手順4を省いた場合は重み
#    ダウンロードも走る:
.venv/bin/python run.py -i ./inputs/example0.mp4 -s 4 -v 11 -m tiny -d cuda:0 -t bf16 -a sage ./_smoke
```

補足:
- `sageattention` pip パッケージは**不要**。`-a sage` が使うのは fork が同梱する
  `sparse_sage` カーネルで、`sageattention` の import は guard 済み。
- 完了後 `<repo>/models/FlashVSR-v1.1/` に
  `diffusion_pytorch_model_streaming_dmd.safetensors`・`Wan2.1_VAE.pth`・
  `LQ_proj_in.ckpt`・`TCDecoder.ckpt`、隣に `<repo>/models/posi_prompt.pth` が揃う
  ——これが `--flashvsr-repo` の期待する構成。

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
| `--flashvsr-max-clip-frames` | `32` | Phase 1 の `--max-clip-size` を上限化し、各 clip を FlashVSR tiny の VRAM に収める。 |
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

jasna 自身の実測(RTX 5080 16 GB / Linux、`small-01.mp4` 480p / 4930 フレーム・全編
モザイク。全走行で出力フレーム数 = 入力。壁時計はコマンド全体、VRAM は `nvidia-smi` の
GPU 全体ピーク):

| モード | scale / tiles | 壁時計 | FlashVSR 時間 | VRAM ピーク | 備考 |
|---|---|---|---|---|---|
| 一次のみ | — | 26 秒 | — | 3.8 GB | 参照(clip 32 + fp8-recon) |
| inline | 4 / 2 | 853 秒 | 841 秒 | 13.4 GB | |
| inline | 2 / 1 | 175 秒 | 163 秒 | 9.9 GB | 4 / 2 比で **4.9 倍速、3.5 GB 低い** |
| inline、1080p(`test-flashvsr-fhd-02`、4203 f) | 2 / 1 | 283 秒 | 274 秒 | 10.7 GB | offload 0、アロケータ警告 0 |
| offline | 4 | 920 秒 | Phase 2 864 秒 | 13.1 GB | bundle 16 GB(ゲート見積り 14.8 GiB) |
| offline | 2 | 410 秒 | Phase 2 361 秒 | 7.8 GB | bundle 4.3 GB(ゲート見積り 3.7 GiB) |

同じ worker なのに jasna の壁時計が lada-ex の約 2 倍かかるのは clip の切り方の差で、
worker の差ではない。jasna は両モードとも FlashVSR へ渡す clip を 32 フレームに
上限化しており(temporal overlap 8 と FlashVSR が要求する 8n+5 パディングで、worker は
ソース 1 フレームあたり約 1.3 DiT フレームを処理する。本素材で 175 clip)、clip ごとに
tiny-long の呼び出し初期化を払う。lada-ex は 180 フレームの clip を渡す。移植前の
worker との同一 clip での A/B は FlashVSR 時間が一致した(scale 2 で 162 秒 vs
163 秒)ので、移植自体のコストはゼロで、常時適用の色補正が FlashVSR 時間の約 7% を
足す。

### 色補正

FlashVSR が生成したクロップは、元になった一次復元結果から色味がずれることがあり、
ブレンド後に復元領域と周囲の色調差として見える。そのため両モードとも、各出力
クロップを常に**入力クロップ(一次出力)の bicubic 拡大**を参照に補正する。方式は
wavelet 再構成で、FlashVSR 出力の高周波(テクスチャ)を入力の低周波(局所的な
色調)の上に載せる。適用は clip ごとにクロップ全体へ 1 回(短冊ごとには行わない)、
量子化の前で、両モードが同じ関数を使う。

実測は、二次が変更した画素内のチャネル毎 median |Δmean|(8bit、同じ clip 構成の
一次のみ出力に対して)。lada-ex: 補正なし 5.28 → AdaIN 0.98 → **wavelet 0.34**
(scale 2 では 0.32)。jasna の scale 2(480p `small-01`): inline 1.76 → 0.43 →
**0.24**、offline 1.61 → **0.24**(関数を共有する両モードが同じ値に落ちる)。
上流の FlashVSR_plus にも `color_fix` はあるが jasna は使わない。呼び出しが裸の
`except: pass` で包まれており、失敗しても「未適用」と区別が付かないためである。

CLI フラグは無い。A/B 検証専用に、環境変数
`JASNA_FLASHVSR_COLOR_FIX=adain|wavelet|none` で両モードの方式を上書きできる。

### clip 長を上限化する理由

Phase 2 は FlashVSR の **tiny** モードを使う。tiny は全 latent フレームを VRAM に
保持し、ロスレスな tensor を返す。パイロットでは 16 GB カードで 21 フレーム=~13.5 GB、
65 フレームで near-OOM だった。そのため一次の clip 長を上限化し
(`--flashvsr-max-clip-frames`、既定 32)、各 clip をその予算内に収める。これが
FlashVSR モードで clip が通常より短くなる理由で、増える継ぎ目は clip 境界の crossfade
が吸収する。上限を上げると Phase 2 で OOM の恐れがある。
上限は `--flashvsr-scale 2` でも同じ 32 である。scale 2 では tiny の latent が 1/4 に
なり(480p 素材で Phase 2 のピークは scale 4 の 13.1 GB に対し 7.8 GB)、緩和の余地は
あるが、現状のビルドでは行っていない(今後の検討事項)。

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
| 必要 VRAM | 各段が非同時なので実質 tiny 単体分 | primary と**同時常駐**(scale 4: 実測 ~14.8 GB @16 GB カード。scale 2: ~10 GB) |
| FlashVSR checkout | パッチ不要 | **tiny-long マルチチャンク修正のパッチ必須** |
| 段階再開 | 可(bundle 永続化) | 不可(単一パス) |
| 進捗 / キャンセル / GUI | 3 段フロー | 通常 secondary と同じ |

### 前提: tiny-long パッチ

inline は VRAM 定常(O(1))の **tiny-long** を使う。FlashVSR_plus の tiny-long は
第 2 チャンクで壊れる既知バグ(`8192 vs 4096` エラー)があり、**修正パッチを当てた
checkout が必須**。jasna は起動時に checkout を検査し、未パッチなら明示エラーで停止して
`flashvsr`(オフライン、tiny、パッチ不要)を案内する。

パッチ本体は
[`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../../patches/flashvsr_plus_tinylong_multichunk_fix.patch)
に同梱。FlashVSR_plus checkout で当てる:

```bash
cd ~/FlashVSR_plus
git apply /path/to/jasna/patches/flashvsr_plus_tinylong_multichunk_fix.patch
```

やっていることは 2 箇所のチャンク跨ぎキャッシュ clear を無効化するだけ
(`src/pipelines/flashvsr_tiny_long.py` の per-chunk `LQ_proj_in.clear_cache()` と
`TCDecoder.clean_mem()` を削除。ループ前の一度きりのリセットは残す)。

### 挙動と制約

- **clip 32 上限・frame-gen off を強制**(オフラインと同じ理由)。`--max-clip-size` は
  自動的に 32 へ丸められる。
- **fp8-recon を自動有効化**(未指定時)。一次のピークを ~0.9–1.7 GB 下げ、同時常駐の
  予算に収める。GPU が fp8 非対応(sm89 未満 / `--fp16` 無し)なら TRT へフォールバック。
- 同期実行。FlashVSR(~15 crop-fps)が律速なので、モザイクが多い区間はその速度に
  律速される(モザイクの無いフレームは一次のみで高速)。FlashVSR が壁時計を支配する
  ため、`--batch-size` を下げても速度低下はほぼ無い。
- VRAM(**16 GB カード + デスクトップ常駐**時、既定の scale 4): 480p で combined
  ~14.8 GB。ただし **1080p 以上は物理天井際**まで上がる(実測 ~15.8 GB ピーク)。worker の
  `expandable_segments` と jasna の `vram_offloader`(キューフレームを system RAM へ
  退避)が圧を吸収して落ちない(1080p では
  `expandable_segments: memory mapping failed with OOM` の**警告**(無害。クラッシュ
  ではない)と大量の offload が出る)。天井が近いときの第一の対策は
  **`--flashvsr-tiles`**(次節)。補助として `--batch-size 2`(または `1`)や
  MPS 停止(~490 MB 増)もある。VRAM が少ない環境や未パッチ checkout では
  オフライン(`flashvsr`)を使う。**`--flashvsr-scale 2` では様相が変わる**: 分割なしで
  Linux の 480p 9.9 GB、1080p 10.7 GB(offload 0、アロケータ警告 0)で、タイリングも
  天井対策も要らない。全表は「[処理倍率](#処理倍率--flashvsr-scale)」。
- **Windows では `expandable_segments` が使えず worker の reserved が ~13 GB に膨らむ**
  ため、tiles 無しの inline は物理天井に張り付く(完走はするが余裕がほぼ無い)。
  1080p では **`--flashvsr-tiles 2` を推奨**。実測は
  「[Windows での注意事項](#windows-での注意事項)」。

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
  羽根合成、色補正もここ)。このファイルは lada-ex の `flashvsr_worker.py` と
  **バイト単位で同一に保つ**(SeedVR2 worker と同じ方針)。wire が lada ネイティブの
  BGR なのはそのためで、FlashVSR_plus 側の互換破壊は片方で直して diff コピーする。
- CLI 配線: `jasna/main.py`。テスト: `tests/test_flashvsr_inline.py`。

## Windows での注意事項

検証環境: Windows 11 / RTX 5080 16 GB / torch 2.13.0+cu130(FlashVSR venv)。結論:
**16 GB カードでは、オフライン(`flashvsr`)か、inline + `--flashvsr-tiles`
(1080p は `2` 推奨)を使う。tiles 無しの inline は完走はするが余裕がほぼ無い。**

- **PyTorch の `expandable_segments` は Windows 未対応**(警告を出して既定の
  キャッシングアロケータへフォールバック)。tiny-long の reserved VRAM は Linux の
  「フラット ~11.9 GB」より断片化で **+1〜2 GB** 膨らむ。`backend:cudaMallocAsync`
  でも改善しない(実測でむしろ微増)。jasna は Windows では worker に
  `expandable_segments` を設定しない。
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
  転じうる。**1080p の常用は `--flashvsr-tiles 2`**(余裕 ~2 GB、減速 +25%)。
  短冊境界のシーム(バンディング、色調ずれ)はこの実測でも検出されなかった。
- 実測ピーク(scale 4 / tiny-long / bf16 / sage / 85 フレーム、reserved 値):
  - **256px 入力(jasna の実ワークロード): ~13.0 GB** — Phase 2 は GPU を単独占有
    するので、オフラインは 16 GB Windows で動く。
  - **384px 入力(同梱 example0 での smoke): ~15.1 GB** — 空きと紙一重。ブラウザや
    IDE が数百 MB 使っているだけで OOM する。**smoke の OOM ≠ jasna 実負荷の OOM**。
- **`-m tiny`(O(T))での 85 フレーム smoke は 16 GB Windows では OOM して正常**。
  smoke は `-m tiny-long` で行う(セットアップ手順 5 のコマンドの `-m tiny` を
  読み替える)。
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
