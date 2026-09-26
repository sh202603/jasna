# SwiftVR 二次復元(`+modi`)

`--secondary-restoration swiftvr-inline` は、一次復元された 256px のモザイククロップを
[SwiftVR](https://github.com/H-oliday/SwiftVR)(one-step streaming diffusion VSR、
Wan2.2-TI2V-5B バックボーン。jasna は fork
[`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR) を使用)で拡大し、
一次の BasicVSR++ では大きなモザイク領域や接写、4K 素材でぼやけがちなテクスチャの
写実性を補う。位置づけは [FlashVSR 二次復元](flashvsr.md)の inline モードと同じで、
処理解像度もモデルネイティブの 1024px(4x)か `--swiftvr-scale 2` の 512px から選ぶ。
どちらでもブレンドがクロップを元の領域へ縮小合成するため、出力動画の解像度は
変わらない。出力クロップは常に、元になった一次復元結果を参照して色補正される
(「[色補正](#色補正)」)。

FlashVSR inline との違いは速度と VRAM にある。SwiftVR は FP8 と torch.compile を
既定で使い(「[高速化](#高速化--swiftvr-accel)」)、RTX 5080 では 90 フレームの
クロップ 1 clip を scale 4 で約 2 秒、scale 2 で約 0.6 秒で処理する。FlashVSR
tiny-long の同条件(scale 4 は短冊 2、scale 2 は短冊なし)に対して clip 単位で約 4〜6 倍、
実行全体で約 2〜4 倍速い(「[VRAM と速度](#vram-と速度)」)。VRAM は scale 4 でも短冊分割
なしに一次と同時常駐できる。

現時点で SwiftVR にあるのは inline モードだけである。オフライン 3 段
(FlashVSR の `flashvsr` に相当)は inline の検証後に追加する予定で、SeedVR2 一次との
併用や 12 GB 級 GPU はそれまで FlashVSR のオフラインモードで組む。

## 仕組み

`swiftvr-inline` は jasna の通常のストリーミングパイプラインの中で、二次復元段として
SwiftVR を走らせる。restorer が SwiftVR 仮想環境の Python で worker
(`jasna/restorer/swiftvr_inline_worker.py`)を常駐起動し、clip ごとに 256px クロップを
stdin で送って 256×scale px の復元クロップを stdout で受け取る(JSON ヘッダ + 生の
uint8、FlashVSR worker と同じプロトコル)。中間ファイルは作らない。

worker は SwiftVR fork の `SwiftVRPipeline.restore_clip()` を呼ぶ。この API はメモリ上の
clip を受け取り、フレーム数を厳密に保って返す(上流の `restore_video()` はファイル
入出力でフレーム数を 4k+1 に切り詰め、`StreamSession` は先頭でフレームを落とすため、
どちらも二次復元の契約を満たさない)。SwiftVR は clip を固定長の因果チャンク
(先頭 28 フレーム、以降 24 フレーム刻み)に分けて処理するので、VRAM は clip 長に
依存しない。

## 必要なもの

SwiftVR は**同梱していない**。checkout、チェックポイント(約 20 GB)、専用仮想環境を
利用者が用意し、`--swiftvr-repo` で jasna に渡す。

checkout には fork [`sh202603/SwiftVR`](https://github.com/sh202603/SwiftVR)(既定
ブランチ `modi`)を使う。fork は上流に `restore_clip()`、uv パッケージング、FP8 DiT、
torch.compile、cuDNN attention、省 VRAM の ReAE を加えたもので、モデルと
チェックポイントは上流と同じである。上流の checkout には `restore_clip()` が無く、
jasna は起動時にこれを検査して明示エラーで停止する。

### SwiftVR checkout のセットアップ(一度だけ)

```bash
# 1. fork を clone(既定ブランチ modi)。
git clone https://github.com/sh202603/SwiftVR.git ~/SwiftVR
cd ~/SwiftVR

# 2. .venv の作成と依存の導入。torch は pyproject.toml の設定で PyTorch の cu132 index
#    から入る(Python 3.12 以上)。Linux では基底 Python に開発ヘッダ(Python.h)が要る:
#    FP8 の量子化カーネルと torch.compile が使う Triton は、初回に C のランチャを
#    ビルドするためである(system Python なら python3.X-dev を入れる。Windows は
#    triton-windows が同梱するので不要)。ヘッダが無い場合も jasna は動くが、worker が
#    起動時に高速化を外し、bf16 の標準経路(約 12 GiB)になる。
uv sync

# 3. チェックポイント(約 20 GB)。既定の置き場は <repo>/checkpoints で、別の場所に
#    置くなら --swiftvr-model-dir で渡す。
uv run hf download H-oliday/SwiftVR --local-dir checkpoints/

# 4. (推奨)jasna に組み込む前に SwiftVR 単体でスモークテスト。inline が使う
#    FP8 + torch.compile を叩く。
uv run swiftvr --input some.mp4 --output out.mp4 --checkpoint checkpoints/ \
    --upscale 4 --fp8-dit --torch_compile
```

### jasna から指定するもの

- `--swiftvr-repo <path>`(必須): 上で作った checkout。
- `--swiftvr-python <path>`(既定 `<repo>/.venv/bin/python`、Windows は
  `<repo>/.venv/Scripts/python.exe`): 手順 2 の venv の Python。
- `--swiftvr-model-dir <path>`(既定 `<repo>/checkpoints`): チェックポイント。

## 使い方

```bash
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration swiftvr-inline \
      --swiftvr-repo ~/SwiftVR \
      --log-level info
```

### フラグ

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--swiftvr-repo` | (必須) | SwiftVR checkout のパス(fork、`restore_clip()` を持つこと)。 |
| `--swiftvr-python` | `<repo>/.venv/bin/python` | SwiftVR 環境の Python(`uv sync` が作る venv)。 |
| `--swiftvr-model-dir` | `<repo>/checkpoints` | チェックポイントのディレクトリ(`reae.safetensors`、`prompt_embedding.safetensors`、`transformer/`)。 |
| `--swiftvr-scale` | `4` | 処理倍率。`4` = モデルネイティブの 1024px、`2` = 512px(高速、低 VRAM)。詳細は「[処理倍率](#処理倍率--swiftvr-scale)」。 |
| `--swiftvr-accel` / `--no-swiftvr-accel` | **on** | FP8 DiT と torch.compile。RTX 40 系以降と動く Triton が必要で、使えない部品は worker が起動時に外して警告する。詳細は「[高速化](#高速化--swiftvr-accel)」。 |

FlashVSR にある `--flashvsr-version`、`--flashvsr-dtype`、`--flashvsr-tiles`、
`--flashvsr-lora` に相当するものは無い。モデルは 1 種類で bf16 固定(FP8 は bf16 が
前提)、短冊タイリングは要らず、LoRA は持たない。色補正にもフラグは無い(常時適用、
後述)。

### 処理倍率(`--swiftvr-scale`)

SwiftVR は 4x モデルで、256px クロップを倍率ぶん bilinear で前拡大してから DiT が
処理する。`4` は学習時と同じ 1024px 処理で、`2` は 512px 処理である。512px では
DiT のトークン格子が 16×16 で窓が 1 個になり、窓の shift が効かないため、学習時と
attention の構造が違う。動作はするが品質は scale 4 と同じゲートで別途確認した
(「[検証](#検証)」)。数値のゲートは通るが、目視では scale 4 より時間安定性が弱く、
1080p 素材でははっきり分かる。既定の scale 4 を推奨する。どちらでも出力動画の解像度は
変わらない。

scale 2 の揺れは設定では消せない。one-step の生成は出力の画素の尺度でテクスチャを
作るので、scale 2 では生成されたテクスチャとそのフレームごとの揺れが、被写体に対して
scale 4 の 2 倍粗い尺度で見える。チャンク境界、FP8、窓の大きさは原因ではなく、生成を
弱める調整は揺れと同じだけ肌理も減らす(「[scale 2 の時間安定性](#scale-2-の時間安定性)」)。

### 高速化(`--swiftvr-accel`)

既定で on。fork の 2 つの高速化部品を使う。

- **FP8 DiT**: DiT ブロックの線形層を FP8 の GEMM で実行する。RTX 40 系以降
  (compute capability 8.9 以上)限定。DiT の重みが 9.4 GiB から 4.8 GiB に減り、
  ピーク VRAM が bf16 の約 12 GiB から約 8 GiB に下がる。
- **torch.compile**: DiT の要素ごとの演算を融合する。初回の warmup で 4 つのグラフ
  (チャンク種別 2 × 窓 shift の有無)をコンパイルするので起動が数秒延び、以後は
  クロップが常に 256px なので再コンパイルは起きない。

fork の計測(RTX 5060 Ti、640×480 → 1280×960)では、両方で GPU スループット 9.1 →
23.3 fps、ピーク 12.0 → 8.0 GiB。出力は bf16 とわずかに異なる(FP8 の丸めを DiT が
増幅し、bf16 比で約 47 dB。fork の目視 A/B では差は見えなかった)。

どちらの部品も Triton を使う。worker はモデルを読む前に、GPU の世代と Triton の
動作(小さなカーネルを 1 回実行)を確かめ、使えない部品を外して理由を jasna のログに
出す。FP8 が外れると DiT は bf16(約 12 GiB)で動き、**16 GB カードでは一次と同居
できない**。この場合 jasna は警告を出して続行し、VRAM が足りなければ clip の処理が
OOM で失敗する(worker は 1 回やり直し、再失敗で停止)。24 GB 以上のカードなら
`--no-swiftvr-accel` でも動く。

### 色補正

SwiftVR が生成したクロップも、元になった一次復元結果から色味がずれることがあり、
ブレンド後に復元領域と周囲の色調差として見える。そのため各出力クロップを常に
**入力クロップ(一次出力)の bicubic 拡大**を参照に補正する。方式は FlashVSR と同じ
wavelet 再構成(SwiftVR 出力の高周波を入力の低周波の上に載せる)で、関数も
FlashVSR worker のものを path 読み込みで共有する。SwiftVR の出力は GPU 上にあるので、
補正は GPU 上でフレームごとに適用する(FlashVSR worker の host 往復版と数値は同じ)。

CLI フラグは無い。A/B 検証専用に、環境変数 `JASNA_SWIFTVR_COLOR_FIX=adain|wavelet|none`
で方式を上書きできる。シェルに設定したまま戻し忘れると以後の走行が全て上書き値で
回るので、`JASNA_SWIFTVR_COLOR_FIX=none jasna ...` のようにコマンド単位で渡すこと。

### clip 長と短いクリップ

clip の長さは通常どおり `--max-clip-size`(既定 90)で決まり、上限は無い。SwiftVR は
clip を先頭 28 フレーム、以降 24 フレーム刻みの固定チャンクに分けて処理するので、
VRAM は clip 長に対して平坦である(90 フレームで 4 回の DiT 呼び出し)。

短い clip は `restore_clip()` の中で最後のフレームの複製で埋める。埋める先は、
チャンクプロトコルの 4k+1 と、**25 フレーム以上**の両方である。28 フレーム以下の clip は
LAST チャンク 1 個になり、DiT への入力は常に 7 latent 分なので、2 フレームでも 25
フレームでも DiT のコストは同じである。違いは埋め方で、25 まで埋めると 7 latent の
すべてが(複製を含む)実フレームから作られ、それより短いとゼロの latent で埋まる。
25 埋めはゼロ latent を避ける設計上の選択で、追加コストはオートエンコーダの数フレーム分
だけである。

なお、短い clip の出力はどう埋めても、同じフレームを長い clip の中で処理した結果とは
異なる(DiT はチャンク内の全フレームを見るので、後続フレームが複製かゼロかで結果が
変わる)。RTX 5080 の実測では、1〜21 フレームの clip を単独で処理した結果は 89 フレーム
の clip 内の結果に対して 31〜37 dB で、ゼロ埋めと 25 埋めのどちらが近いかは clip 長に
よってまちまちだった(25 フレームの実フレームがある場合は 47 dB)。数値で優劣が
つかないため、ゼロ latent を生まない 25 埋めを採る。FlashVSR も同様に 21 フレームまで
複製で埋めており、短い clip の見え方は目視で確かめる。

## 挙動と制約

- **frame-gen off を強制**(FlashVSR inline と同じ理由)。
- **fp8-recon を自動有効化**(未指定時)。一次のピークを ~0.9〜1.7 GB 下げ、同時常駐の
  予算に収める。GPU が fp8 非対応なら TRT へフォールバック。
- **SeedVR2 一次復元とは併用不可**(常駐 worker 2 つで 16 GB を超える。起動時エラー)。
- 同期実行。SwiftVR が律速なので、モザイクが多い区間はその速度になる(モザイクの
  無いフレームは一次のみで高速)。
- VR モードと `--stream` は FlashVSR inline と同じく拒否しない。
- worker の起動には、チェックポイントの読込(約 20 GB、ページキャッシュに乗っていれば
  数秒)と warmup(torch.compile 込みで数秒)がかかる。ハンドシェイクの待ち時間は
  600 秒で打ち切る。
- **同梱なし / サポーターモデルとは無関係**。SwiftVR は Apache-2.0 のサードパーティ
  モデルで、checkout、チェックポイント、venv は利用者が用意する。GUI には出ない。

## 検証

### SwiftVR 側(`restore_clip()`)

RTX 5080(16 GB)、Linux、fork の 256px 89 フレームのクリップで確認した。

- `runner.py` のチャンク処理の切り出し前後で、`restore_video()` の PNG 出力は
  bf16、FP8 + torch.compile(4x)、FP8 + torch.compile(2x)の 3 構成ともビット一致。
- 同じ 89 フレームを `restore_clip()` に渡した結果は、3 構成とも `restore_video()`
  とビット一致(4k+1 フレームなのでチャンク分割が同じ)。
- 90 フレームの clip 1 個の処理時間(warmup 後の中央値)とピーク VRAM(allocated /
  reserved):

| 構成 | 1 clip | ピーク VRAM |
|------|--------|-------------|
| scale 4、FP8 + compile | 1.58 秒(57 fps) | 7.9 / 8.6 GiB |
| scale 2、FP8 + compile | 0.41 秒(217 fps) | 5.6 / 6.1 GiB |
| scale 4、bf16(`restore_video`、89 フレーム) | 4.2 秒 | 12.1 / 12.6 GiB |

### worker 経由

jasna の restorer から実 worker を起動して確かめた(乱数クロップ、色補正込み、
wire の転送込み)。起動は約 7 秒(モデル読込 2 秒、warmup 4 秒)。90 フレーム clip は
scale 4 で 2.1 秒、scale 2 で 0.6 秒、bf16(`--no-swiftvr-accel`)の scale 4 で 3.9 秒。

### e2e

実素材での完走、GPU 全体のピーク、壁時計、色補正ゲートは
「[VRAM と速度](#vram-と速度)」に記す。目視 A/B(FlashVSR との比較)は利用者が行う。

## VRAM と速度

Linux、RTX 5080 16 GB(デスクトップ常駐 約 1.9 GB を含む GPU 全体のピーク、
`nvidia-smi` の 1 秒ポーリング)、既定の clip 90、`--fp8-recon`(自動有効)、色補正
wavelet。FlashVSR の行は同じ日に同じ素材で取った(fork、`--flashvsr-accel`、scale 4 は
`--flashvsr-tiles 2`)。出力フレーム数は全走行で入力と一致した。

| 素材 | 構成 | 壁時計 | GPU 全体ピーク |
|------|------|--------|----------------|
| 480p、4930 フレーム | 一次のみ | 25.0 秒 | 3.8 GB |
| 480p | `swiftvr-inline` scale 4 | **102.2 秒** | **12.7 GB** |
| 480p | `swiftvr-inline` scale 2 | **44.8 秒** | **10.4 GB** |
| 480p | `swiftvr-inline` scale 4、色補正 none | 97.7 秒 | 12.8 GB |
| 480p | `flashvsr-inline` scale 4 tiles 2 | 411.7 秒 | 14.4 GB |
| 480p | `flashvsr-inline` scale 2 | 100.7 秒 | 10.6 GB |
| 1080p、4203 フレーム | `swiftvr-inline` scale 4 | **139.5 秒** | **14.0 GB** |
| 1080p | `swiftvr-inline` scale 2 | **56.9 秒** | **12.0 GB** |

- 実行全体では FlashVSR inline に対して scale 4 で約 4 倍、scale 2 で約 2.2 倍速い。
  一次のみの 25 秒を除いた二次の分では、scale 4 で約 5 倍(387 → 77 秒)、scale 2 で
  約 3.8 倍(76 → 20 秒)。1080p の FlashVSR は以前の実測で scale 2 が 127 秒 / 10.1 GB、
  scale 4 tiles 2 が 556 秒 / 13.8 GB。
- scale 4 は 1080p で 14.0 GB と天井まで 2 GB 強。短冊分割は無い。scale 2 は
  VRAM に余裕があるが、次項の目視で時間安定性が劣るので、VRAM が許す限り scale 4 を使う。
- 色補正のコストは壁時計の約 4.6%(102.2 秒 対 97.7 秒)。
- `--no-swiftvr-accel`(bf16、480p、scale 4): 警告を出して続行し、最初の clip で
  worker が OOM(1 回やり直して clip エラー)、二次スレッドの例外で実行は 13 秒後に
  終了コード 1 で止まった(ハングなし。GPU 全体 14.6 GB)。設計どおり。

### ゲート

- **色ずれ**(二次が変更した画素内のチャネル毎 median |Δmean|、一次のみ出力比、480p、
  `scripts/evaluation/flashvsr-color-fix-report.py`): 補正なし 0.765 → **wavelet
  scale 4 0.225、scale 2 0.241**(合格)。FlashVSR の同じ指標は 1.73 → 0.25 で、
  SwiftVR は補正なしのずれが小さく、補正後は同水準。
- **時間方向の変化**(二次が変更した領域内の隣接フレーム差の、一次のみ出力に対する比。
  flow 補償なしの粗い代理指標): 480p では scale 4 で 1.021、scale 2 で 1.056、補正なしで
  1.030。1080p(4 フレームごとの対)では scale 4 で 1.32、scale 2 で 1.50 と大きく、
  目視の印象(次項)と向きが一致する。
- **目視 A/B**(利用者、480p と 1080p、FlashVSR inline の同素材出力と比較): 復元の
  肌理とディテールは scale 4 で FlashVSR と同等。**scale 2 は scale 4 より時間安定性が
  弱く、1080p 素材でははっきり視認できる**(上の代理指標でも scale 2 のほうが大きい)。
  したがって既定の scale 4 を使い、scale 2 は速度優先で時間方向の揺らぎを許容できる
  場合に限る。

### scale 2 の時間安定性

scale 2 の揺れの原因を、1080p 素材(4203 フレーム)の一次復元済みクロップ 58 clip
(5085 フレーム)で切り分けた。SwiftVR の `restore_clip()` を直接回し、出力に
production と同じ wavelet 色補正をかけてから、768px で一次のみのクロップ(bicubic で
拡大)と比べた。指標は flow-warping error 比(flow は一次のみのクロップから SPyNet で
一度だけ計算し、両方に同じ flow を使う。低いほど時間方向に安定)と、鮮鋭さ
(ラプラシアン分散の比)である。

| 変種 | flow-warping error 比 | 鮮鋭さ |
| --- | --- | --- |
| scale 4 | 1.89 | 24.2 |
| scale 2 | 3.88 | 18.3 |
| scale 2、128px に縮小してから 4 倍 | 3.11 | 11.9 |

- **チャンク境界ではない**: チャンク境界(出力フレーム 25、49、73)をまたぐ対と
  それ以外の対で flow-warping error 比が同じ(512px 評価で 3.758 と 3.763)。
  前チャンクの latent を文脈に足す overlap(1、2 latent)と、チャンク長 48 は改善しない。
- **FP8 ではない**: bf16 でも 3.75 で変わらない(512px 評価、FP8 は 3.76)。
- **窓の大きさではない**: 窓を 8×8 にして 512px でも shift を効かせても改善しない。
- **DiT の生成そのもの**: DiT を通さず TAE で往復するだけなら scale 2 でも 1.26 と
  安定している(先頭 20 clip)。
- **生成を弱めると肌理も減る**: 入力の縮小やぼかし、DiT の予測の縮小、timestep を
  下げる調整は、いずれも揺れを減らした分だけ鮮鋭さを落とし、scale 4 の水準には届かない。
  scale 4 は安定性と鮮鋭さの両方でこれらを上回る。

以上から scale 2 は現状のまま据え置き、scale 4 を推奨する。

## 実装

- `jasna/restorer/swiftvr_common.py`: `--swiftvr-*` の登録とパスの解決(torch を
  import しない)。
- `jasna/restorer/swiftvr_inline_secondary_restorer.py`: 同期 `SecondaryRestorer`。
  worker の起動とハンドシェイク、wire の入出力、RGB と BGR の反転、keep window の切り出し。
  FlashVSR inline restorer と同じ構造で、パッチ検査、高速化の環境変数、実行中の降格報告と
  respawn を持たない。
- `jasna/restorer/swiftvr_inline_worker.py`: SwiftVR venv で動く worker。jasna も
  lada も import しない(lada-ex にそのまま持ち込める)。高速化の判定、モデル読込、
  warmup、clip ごとの `restore_clip()` と色補正。色補正の primitive は
  `flashvsr_inline_worker.py` を path 読み込みで共有する。
- `jasna/session_config.py` / `session_factory.py` / `main.py`: 設定フィールド、
  restorer の生成、起動時検査(fp8-recon 自動有効化、frame-gen と SeedVR2 一次の拒否)。
- `scripts/build_nuitka.py`: worker を実ファイルとして `<dist>/jasna/restorer/` に複製する
  (FlashVSR worker と並べて置く。色補正の共有のため)。
- テスト: `tests/test_swiftvr_inline.py`(stub worker で wire、フラグ、ハンドシェイク、
  色補正の GPU 版と FlashVSR 版の一致)、`tests/test_main.py`(choices と既定値)。

SwiftVR fork 側: `swiftvr/runner.py` の `restore_chunk()`(オフライン runner と共有)、
`swiftvr/pipeline.py` の `restore_clip()`。

## Windows での注意事項

Windows では未検証である。想定される差は次のとおり。

- `expandable_segments` が使えないので worker の reserved が Linux より増えるが、
  SwiftVR fork の FP8 の実測(8.0 GiB)は Windows 機のものなので、大きな差は出ない見込み。
- Triton は `triton-windows` が `uv sync` で入り、C++ コンパイラは要らない(fork の
  README)。
- `--swiftvr-python` の既定は `<repo>/.venv/Scripts/python.exe`。
