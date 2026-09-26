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

SwiftVR にあるのは inline モードだけである。オフライン 3 段(FlashVSR の `flashvsr` に
相当)は無いので、SeedVR2 一次との併用や 12 GB 級 GPU は FlashVSR のオフラインモードで
組む。

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

worker に渡すクロップは、一次復元の出力そのものではなく、その配置を時間方向に
平滑化した **view** である。一次復元のクロップは検出枠に追従して毎フレーム位置が動き、
SwiftVR はその動きに反応してディテールを描き直すため、一次段の末尾で各フレームの
256px 格子を前後 15 フレームの平均の配置に再サンプルしてから送り、blend は SwiftVR の
出力を view の配置で元フレームへ直接合成する(「[切り出し view の平滑化](#切り出し-view-の平滑化--swiftvr-view-window)」)。

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
| `--swiftvr-view-window` | `15` | SwiftVR に渡す切り出し(view)の位置と倍率を前後 N フレームの移動平均で平滑化する(`0` で無効)。詳細は「[切り出し view の平滑化](#切り出し-view-の平滑化--swiftvr-view-window)」。 |
| `--swiftvr-accel` / `--no-swiftvr-accel` | **on** | FP8 DiT と torch.compile。RTX 40 系以降と動く Triton が必要で、使えない部品は worker が起動時に外して警告する。詳細は「[高速化](#高速化--swiftvr-accel)」。 |

FlashVSR にある `--flashvsr-version`、`--flashvsr-dtype`、`--flashvsr-tiles`、
`--flashvsr-lora` に相当するものは無い。モデルは 1 種類で bf16 固定(FP8 は bf16 が
前提)、短冊タイリングは要らず、LoRA は持たない。色補正にもフラグは無い(常時適用、
後述)。

## 処理の詳細

### 処理倍率(`--swiftvr-scale`)

SwiftVR は 4x モデルで、256px クロップを倍率ぶん bilinear で前拡大してから DiT が
処理する。`4` は学習時と同じ 1024px 処理で、`2` は 512px 処理である。どちらでも出力動画の
解像度は変わらない。

scale 2 は scale 4 に対して実行全体で約 2.5 倍速く、GPU 全体のピークが 2〜2.5 GB 低い
(「[VRAM と速度](#vram-と速度)」)。品質は、切り出し view の平滑化(既定で有効)の下で
目視でも scale 4 と同等で、時間安定性、質感、位置の正確さ、clip 境界の見え方に差は
出ない(「[品質のゲート](#品質のゲート)」)。既定は学習時と同じ scale 4 で、速度と
VRAM を優先するなら scale 2 を使う。

scale 2 と scale 4 の残る違いは生成の構造にある。512px では DiT のトークン格子が
16×16 で窓が 1 個になり、窓の shift が効かないため、学習時と attention の構造が違う。
また one-step の生成は出力の画素の尺度でテクスチャを作るので、scale 2 の生成は
被写体に対して scale 4 の 2 倍粗い尺度で行われる。この違いは設定では変わらず、
生成を弱める調整(入力の縮小やぼかし、DiT の予測の縮小、timestep を下げる)は揺れを
減らした分だけ肌理も減らす(「[時間安定性を決める要因](#時間安定性を決める要因)」)。

### 切り出し view の平滑化(`--swiftvr-view-window`)

既定で有効(15 フレーム、`0` で無効)で、scale 2 と 4 の両方に効く。

一次復元のクロップは検出枠に追従する。jasna は clip 内の最大枠に合わせた共通倍率で
クロップを縮小し、256px 格子の中央に置いて鏡で埋めるので、被写体の格子上の位置は
フレームごとに動く(1080p 素材の実測で平均 3.0 格子 px/frame、枠幅の変化は最大 18%)。
SwiftVR はサブピクセル級の入力のずれにも反応してディテールを描き直し、しかも出力は
入力のずれに半分程度しか追従しない(位相相関で毎フレーム 1.2〜1.6 px@512 のずれ)。
そのままでは復元領域が周囲に対して毎フレーム位置を変え、輪郭の太さと質感が
フレームごとに揺れて見える。scale 2 ではこれが 1080p 素材ではっきり視認できる。

平滑化は SwiftVR に渡す入力の配置だけを安定させ、出力を正確に元の位置へ戻す。
一次復元と blend の mask、worker と wire は変わらない。

- 一次段の末尾(GPU 上)で、各フレームの 256px 格子を、前後 N フレームの移動平均で
  求めた配置(view)へ双線形で再サンプルする。格子の外は鏡で埋める。
- 平滑化した view がそのフレームのクロップを覆えない場合(窓 15 で全フレームの
  約 10%、はみ出しは中央値 5 px、最大 82 px)は、view の位置を必要最小限だけずらして
  覆う。このクランプ後も枠のずれは平滑化なしの半分以下に収まる(3.04 → 1.45 格子
  px/frame)。
- blend 段は SwiftVR の出力(view 格子)から元フレームの画素位置へ直接 1 回で
  リサンプルする。配置が元の格子と同じならこれは従来の合成と同じ位置をサンプルするので、
  無効時と他の二次復元の経路は変わらない。

窓幅は 15 以上で効果がほぼ飽和し(crop 単位の flow-warping error 比は窓 7 で 2.38、
15 で 2.24、31 で 2.16)、広げるほどクランプが効くフレームが増えるので、15 を既定に
している。格子の外を元フレームの周辺で埋める方式は、当方の指標では鏡余白と差が小さく
(2.24 対 2.34)、二次段に元フレームを渡す経路が要るので採っていない。

効果は crop 単位の flow-warping error 比で scale 2 が 4.99 → 2.57(scale 4 は 2.16)、
目視では scale 2 の時間安定性と質感が scale 4 と同等になる(「[品質のゲート](#品質のゲート)」、
「[時間安定性を決める要因](#時間安定性を決める要因)」)。速度と VRAM への影響は clip
あたり `grid_sample` 1 回分で、実測では誤差の範囲(「[VRAM と速度](#vram-と速度)」)。

平滑化の後も残るのは、SwiftVR の出力に固有の 2 フレーム周期の低域の変動(TAE の
時間圧縮の性質)と、生成の尺度の違い(scale 2 は被写体に対して 2 倍粗い)である。

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

## 実測

Linux、RTX 5080 16 GB。GPU 全体のピークはデスクトップ常駐(約 1.9 GB)を含み、
`nvidia-smi` の 1 秒ポーリングの最大値。既定の clip 90、`--fp8-recon`(自動有効)、
色補正 wavelet、view 平滑化は既定(15)。素材は 480p(4930 フレーム)と 1080p
(4203 フレーム)で、出力フレーム数は全走行で入力と一致した。

### VRAM と速度

| 素材 | 構成 | 壁時計 | GPU 全体ピーク |
|------|------|--------|----------------|
| 480p | 一次のみ | 25.0 秒 | 3.8 GB |
| 480p | `swiftvr-inline` scale 4 | **102.2 秒** | **12.7 GB** |
| 480p | `swiftvr-inline` scale 2 | **44.2 秒** | **10.3 GB** |
| 480p | `swiftvr-inline` scale 4、色補正 none | 97.7 秒 | 12.8 GB |
| 480p | `flashvsr-inline` scale 4 tiles 2 | 411.7 秒 | 14.4 GB |
| 480p | `flashvsr-inline` scale 2 | 100.7 秒 | 10.6 GB |
| 1080p | `swiftvr-inline` scale 4 | **137.0 秒** | **13.9 GB** |
| 1080p | `swiftvr-inline` scale 2 | **54.1 秒** | **11.2 GB** |

- 実行全体では FlashVSR inline に対して scale 4 で約 4 倍、scale 2 で約 2.3 倍速い。
  一次のみの 25 秒を除いた二次の分では、scale 4 で約 5 倍(387 → 77 秒)、scale 2 で
  約 4 倍(76 → 19 秒)。1080p の FlashVSR は以前の実測で scale 2 が 127 秒 / 10.1 GB、
  scale 4 tiles 2 が 556 秒 / 13.8 GB。
- scale 4 は 1080p で 13.9 GB と天井まで 2 GB 強。短冊分割は無い。
- view 平滑化のコストは測定誤差の範囲。`--swiftvr-view-window 0` は 1080p scale 2 で
  53.8 秒 / 11.2 GB、480p scale 2 で 43.8 秒 / 10.3 GB(既定との差は 0.3〜0.4 秒)。
  FlashVSR との比較行と 480p scale 4 の行は平滑化なしの走行で、差は同じく誤差の範囲。
- 色補正のコストは壁時計の約 4.6%(102.2 秒 対 97.7 秒)。
- `--no-swiftvr-accel`(bf16、480p、scale 4): 警告を出して続行し、最初の clip で
  worker が OOM(1 回やり直して clip エラー)、二次スレッドの例外で実行は 13 秒後に
  終了コード 1 で止まった(ハングなし。GPU 全体 14.6 GB)。設計どおり。

### 品質のゲート

- **色ずれ**(二次が変更した画素内のチャネル毎 median |Δmean|、一次のみ出力比、480p、
  `scripts/evaluation/flashvsr-color-fix-report.py`): 補正なし 0.765 → **wavelet
  scale 4 0.225、scale 2 0.224**(合格)。FlashVSR の同じ指標は 1.73 → 0.25 で、
  SwiftVR は補正なしのずれが小さく、補正後は同水準。
- **時間方向の変化**(二次が変更した領域内の隣接フレーム差の、一次のみ出力に対する比。
  flow 補償なしの粗い代理指標): 480p では scale 4 で 1.021、scale 2 で 1.029。1080p
  (4 フレームごとの対)では scale 4 で 1.33、scale 2 で 1.43。view 平滑化なしの
  scale 2 は 480p 1.054、1080p 1.50 で、平滑化で scale 4 に近づくが並びはしない。
  この指標は被写体の動きに伴うディテールの変化と、平滑化と無関係に残る 2 フレーム
  周期の変動も拾うので、crop 単位の flow-warping error 比ほどは下がらない。
- **目視 A/B**(利用者、480p と 1080p): scale 4 の肌理とディテールは FlashVSR inline の
  同素材出力と同等。scale 2 は時間安定性と質感が scale 4 と同等で、位置ずれと clip
  境界(74 フレームごと)の継ぎ目は見えない。scale 4 に view 平滑化を入れても質感は
  落ちない。view 平滑化なしの scale 2 は scale 4 より時間安定性が弱く、1080p 素材では
  はっきり視認できる。

### 時間安定性を決める要因

scale 2 の揺れの要因を、1080p 素材の一次復元済みクロップ 58 clip(5085 フレーム)で
切り分けた。SwiftVR の `restore_clip()` を直接回し、出力に production と同じ wavelet
色補正をかけてから、一次のみのクロップ(bicubic で拡大)と比べる。指標は
flow-warping error 比(flow は一次のみのクロップから SPyNet で一度だけ計算し、両方に
同じ flow を使う。低いほど時間方向に安定)で、512px 評価、クロップの有効域に限定した。

| 変種 | flow-warping error 比 |
| --- | --- |
| scale 2 | 4.99 |
| scale 2、view 平滑化(鏡余白 + クランプ、本番の構成) | 2.57 |
| scale 2、view 平滑化(余白を元フレームの周辺で埋める、クランプなし) | 2.24 |
| scale 4 | 2.16 |
| scale 4、view 平滑化(余白を元フレームの周辺で埋める、クランプなし) | 1.59 |

- **入力の配置が主因**: ぼけだけを揃えた対照(出力を平滑化と同じサブピクセル量だけ
  ずらす)は 4.23 で、2.2〜2.6 への低下は入力の安定化そのものによる。鮮鋭度は、評価の
  ために出力を元の格子へ戻す双線形リサンプルでラプラシアン分散が半減するが、本番の
  blend はこの経路を通らず、view 空間では 324 → 323 と維持される。
- **DiT の生成が揺れの発生源**: DiT を通さず TAE で往復するだけなら scale 2 でも
  安定している(比 1.26、先頭 20 clip、768px 評価)。生成が入力のずれに反応する。
- **チャンク境界ではない**: チャンク境界(出力フレーム 25、49、73)をまたぐ対と
  それ以外の対で比が同じ(3.758 と 3.763)。前チャンクの latent を文脈に足す overlap
  (1、2 latent)と、チャンク長 48 は改善しない。
- **FP8 ではない**: bf16 でも 3.75 で変わらない(FP8 は 3.76)。
- **窓の大きさではない**: 窓を 8×8 にして 512px でも shift を効かせても改善しない。
- **生成を弱めると肌理も減る**: 入力の縮小(128px にしてから 4 倍で 3.11)やぼかし、
  DiT の予測の縮小、timestep を下げる調整は、いずれも揺れを減らした分だけ鮮鋭さを
  落とし、scale 4 の水準には届かない。

### SwiftVR 側の確認

fork の `restore_clip()` を RTX 5080 で確かめた(256px 89 フレームのクリップ)。

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

jasna の restorer から実 worker を起動した場合(乱数クロップ、色補正込み、wire の
転送込み)、起動は約 7 秒(モデル読込 2 秒、warmup 4 秒)。90 フレーム clip は scale 4
で 2.1 秒、scale 2 で 0.6 秒、bf16(`--no-swiftvr-accel`)の scale 4 で 3.9 秒。

## 実装

- `jasna/restorer/swiftvr_common.py`: `--swiftvr-*` の登録とパスの解決(torch を
  import しない)。
- `jasna/restorer/swiftvr_inline_secondary_restorer.py`: 同期 `SecondaryRestorer`。
  worker の起動とハンドシェイク、wire の入出力、RGB と BGR の反転、keep window の切り出し。
  view 平滑化の窓幅を `view_smoothing_window` 属性で申告する。FlashVSR inline restorer
  と同じ構造で、パッチ検査、高速化の環境変数、実行中の降格報告と respawn を持たない。
- `jasna/restorer/swiftvr_inline_worker.py`: SwiftVR venv で動く worker。jasna も
  lada も import しない(lada-ex にそのまま持ち込める)。高速化の判定、モデル読込、
  warmup、clip ごとの `restore_clip()` と色補正。色補正の primitive は
  `flashvsr_inline_worker.py` を path 読み込みで共有する。
- `jasna/tracking/crop_view.py`: 切り出し view の幾何(配置の計算、平滑化、
  クランプ、view への再サンプル、blend のサンプル)。`restorer/restoration_pipeline.py`
  が二次 restorer の `view_smoothing_window` を見て一次段の末尾で view を作り、
  `pipeline_items.py` の `view_placements` で配置を運び、`blend_buffer.py` が配置の
  あるときだけ view から直接合成する(0 または属性なしで従来経路。FlashVSR inline にも
  同じ機構を足せる)。
- `jasna/session_config.py` / `session_factory.py` / `main.py`: 設定フィールド、
  restorer の生成、起動時検査(fp8-recon 自動有効化、frame-gen と SeedVR2 一次の拒否)。
- `scripts/build_nuitka.py`: worker を実ファイルとして `<dist>/jasna/restorer/` に複製する
  (FlashVSR worker と並べて置く。色補正の共有のため)。
- テスト: `tests/test_swiftvr_inline.py`(stub worker で wire、フラグ、ハンドシェイク、
  色補正の GPU 版と FlashVSR 版の一致)、`tests/test_main.py`(choices と既定値)、
  `tests/test_crop_view.py`(view の幾何: 配置が元の格子なら従来経路と一致、平滑化した
  view の往復、クランプの被覆)、`test_restoration_pipeline.py` と `test_blend_buffer.py`
  の view 経路。

SwiftVR fork 側: `swiftvr/runner.py` の `restore_chunk()`(オフライン runner と共有)、
`swiftvr/pipeline.py` の `restore_clip()`。

## Windows での注意事項

Windows では未検証である。想定される差は次のとおり。

- `expandable_segments` が使えないので worker の reserved が Linux より増えるが、
  SwiftVR fork の FP8 の実測(8.0 GiB)は Windows 機のものなので、大きな差は出ない見込み。
- Triton は `triton-windows` が `uv sync` で入り、C++ コンパイラは要らない(fork の
  README)。
- `--swiftvr-python` の既定は `<repo>/.venv/Scripts/python.exe`。
- view 平滑化は jasna 側の GPU 処理だけなので、Windows 固有の要素は無い。
