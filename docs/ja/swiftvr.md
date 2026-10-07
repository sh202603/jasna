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

モードは 2 つある。`swiftvr-inline` は SwiftVR を一次パイプラインと同居させる単一パスで、
16 GB カードの `basicvsrpp` 一次向け。`swiftvr` は FlashVSR の `flashvsr` と同じオフライン
3 段で、SwiftVR を GPU 単独で走らせる(「[オフライン 3 段](#オフライン-3-段--secondary-restoration-swiftvr)」)。
12 GB 級 GPU、FP8 が使えない GPU、SeedVR2 一次との併用はこちらで組む。以下の inline の
記述(処理の詳細、品質)はオフラインにもそのまま当てはまり、違いは「オフライン 3 段」の節に
まとめる。

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

# オフライン 3 段(12 GB 級 GPU、FP8 非対応 GPU、SeedVR2 一次との併用)
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration swiftvr \
      --swiftvr-repo ~/SwiftVR \
      --swiftvr-bundle-dir /data/jasna_bundle \
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
| `--swiftvr-bundle-dir` | temp | `swiftvr`(オフライン)専用。中間 bundle をここに永続化する(段階再開が可能に)。 |
| `--swiftvr-keep-bundle` | off | `swiftvr`(オフライン)専用。完了後も bundle を残す(`--swiftvr-bundle-dir` 指定時は暗黙的に有効)。 |

FlashVSR にある `--flashvsr-version`、`--flashvsr-dtype`、`--flashvsr-tiles`、
`--flashvsr-lora`、`--flashvsr-max-clip-frames` に相当するものは無い。モデルは 1 種類で
bf16 固定(FP8 は bf16 が前提)、短冊タイリングは要らず、LoRA は持たず、VRAM が clip 長に
依存しないので clip 上限も要らない。色補正にもフラグは無い(常時適用、後述)。

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
この原因の特定と対策は [mioh-labs](https://github.com/mioh-labs/mioh) の報告による
(「[謝辞](#謝辞)」)。

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
`--no-swiftvr-accel` でも動く。Windows の 16 GB カードでは OOM にならず完走した実測がある
(約 2 倍遅く、VRAM は上限に張り付く。「[Windows での注意事項](#windows-での注意事項)」)。
オフラインの `swiftvr` では SwiftVR が GPU を独占するので
bf16 が正規の経路で、16 GB に収まる(「[オフライン 3 段](#オフライン-3-段--secondary-restoration-swiftvr)」)。

### 色補正

SwiftVR が生成したクロップも、元になった一次復元結果から色味がずれることがあり、
ブレンド後に復元領域と周囲の色調差として見える。そのため各出力クロップを常に
**入力クロップ(一次出力)の bicubic 拡大**を参照に補正する。方式は FlashVSR と同じ
wavelet 再構成(SwiftVR 出力の高周波を入力の低周波の上に載せる)で、関数も
FlashVSR worker と同じもの(内蔵、bit 一致をテストで確認)である。SwiftVR の出力は GPU 上にあるので、
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
- **SeedVR2 一次復元とは併用不可**(常駐 worker 2 つで 16 GB を超える。起動時エラー。
  オフラインの `swiftvr` なら併用できる)。
- 同期実行。SwiftVR が律速なので、モザイクが多い区間はその速度になる(モザイクの
  無いフレームは一次のみで高速)。
- VR モードと `--stream` は FlashVSR inline と同じく拒否しない。
- worker の起動には、チェックポイントの読込(約 20 GB、ページキャッシュに乗っていれば
  数秒)と warmup(torch.compile 込みで数秒)がかかる。ハンドシェイクの待ち時間は
  600 秒で打ち切る。
- **同梱なし / サポーターモデルとは無関係**。SwiftVR は Apache-2.0 のサードパーティ
  モデルで、checkout、チェックポイント、venv は利用者が用意する。GUI には出ない。

## オフライン 3 段(`--secondary-restoration swiftvr`)

`swiftvr` は FlashVSR の `flashvsr` と同じオフライン 3 段で、各段を別プロセスで順に
走らせ、SwiftVR の段に GPU を独占させる。inline と同じクロップが同じ関数を通るので
出力は inline と一致し(「[オフラインの実測](#オフラインの実測)」)、違いは VRAM の要件、
中間ファイル、再開の可否である。

| 段 | 環境 | 内容 |
|----|------|------|
| 1 (dump) | jasna | decode + detect + 一次復元(BasicVSR++ または SeedVR2)。各 clip の 256px クロップ(inline と同じ切り出し view)+ マスク + 幾何をディスク上の **bundle** へ直列化。blend/encode は捨てる。 |
| 2 (SwiftVR) | SwiftVR venv | 各 clip の 256px クロップを 256×scale px に復元し、色補正して bundle へ書き戻す(`jasna/restorer/swiftvr_phase2_driver.py`)。 |
| 3 (reblend) | jasna | source を再デコードし、bundle から復元結果を再構成、view の配置で再 blend して最終出力を encode。 |

Phase 1 と Phase 3 は FlashVSR と共通のコード(`jasna/restorer/flashvsr_offline.py`)で、
`jasna --flashvsr-phase dump` / `reblend` のサブプロセスとして走る(内部名は `flashvsr` の
まま)。Phase 2 の driver は SwiftVR venv の Python で動き、inline worker
(`swiftvr_inline_worker.py`)を path で読んで、高速化の判定、モデル読込、warmup、
`restore_clip()` の枚数検査付き呼び出し、GPU 上の色補正を共有する。wire が無いので BGR の
反転も無い(bundle は RGB)。

**bundle** の形式は FlashVSR と同じで、version 2 になった。Phase 1 は inline と同じ
切り出し view(`--swiftvr-view-window`、既定 15)を作り、その配置を clip の幾何に
`view_placements` として書く。Phase 3 は配置があれば view から元フレームへ直接合成する
(inline の blend と同じ経路)。FlashVSR の bundle は配置が null で、従来どおり動く。
Phase 3 は version 1 の bundle も読み、より新しい version は拒否する。

### 使いどころ

inline は SwiftVR を一次パイプラインと同居させる。FP8 で約 8 GiB、一次と合わせて 16 GB に
収まるが、次の場合には組めない。オフラインはこれらのための経路である。

1. FP8 が使えない GPU(RTX 30 系以前、compute capability 8.9 未満)。DiT が bf16 の
   約 12.4 GiB になり、16 GB でも一次と同居できない。
2. 12 GB 級の GPU。FP8 でも一次と合わせて 12 GB を超える(scale 4 はアプリ分だけで 11 GB
   超、scale 2 も 1080p では境界)。
3. SeedVR2 一次復元との併用。常駐 worker 2 つで 16 GB を超えるため、inline は起動時に
   拒否する。

オフラインで効くかどうかは SwiftVR 単体のピークで決まる。RTX 5080(16 GB、デスクトップ
常駐約 2.0 GB)で、Phase 2 と同じ読込と warmup の後に 90 フレームの 256px クロップを
3 clip 処理して測った(fork e7f186b、FP8 の DiT をブロック単位で読み込む版。それ以前の
fork は読込時のピークが 10.4 GB に達し、12 GB 級では読み込めない)。allocated / reserved
は torch の値、プロセスピークは `nvidia-smi` の値。

| 構成 | 読込時ピーク | 処理中ピーク(allocated / reserved) | プロセスピーク(expandable あり) | 同(なし、Windows の代理) | 90 フレーム 1 clip |
|---|---|---|---|---|---|
| scale 4、FP8 + compile | 5.3 GiB | 7.8 / 8.2 GiB | 8.8 GB | 9.0 GB | 1.6 秒 |
| scale 4、bf16 | 9.4 GiB | 12.4 / 12.6 GiB | 13.3 GB | 14.0 GB(Windows 実測) | 3.4 秒 |
| scale 2、FP8 + compile | 5.3 GiB | 5.6 / 5.9 GiB | 6.4 GB | 6.5 GB | 0.4 秒 |
| scale 2、bf16 | 9.4 GiB | 10.2 / 10.3 GiB | 10.9 GB | 未測定 | 0.8 秒 |

12 GB 級を模擬して torch の割り当て上限を 10.5 GiB と 9.5 GiB に掛けた FP8 の走行は、
scale 4 と scale 2 のどちらも上限 10.5 GiB で完走し、scale 4 は上限 9.5 GiB でも完走した
(ピークは上限なしと同じ)。

- **12 GB 級で FP8 可(RTX 4070、5070 など)**: scale 4 の FP8 が 8.8 GB で収まり、
  デスクトップ常駐が 3 GB あっても届く。scale 2 は 6.4 GB。
- **RTX 30 系 16 GB**: scale 4 の bf16 が 13.3 GB で収まる(`--no-swiftvr-accel` は不要。
  driver が FP8 を外して bf16 に落ち、警告は出ない)。Windows(`expandable_segments` なし)
  では 14.0 GB で、デスクトップ常駐 1.4 GB 込み 15.4 GB で完走した。常駐が 2 GB を超えると
  天井に届くので、常駐を小さくして走らせる。
- **12 GB 級で FP8 不可(RTX 3060 12 GB など)**: scale 4 の bf16 は入らない。scale 2 の
  bf16 が 10.9 GB で境界。
- **SeedVR2 一次との併用**: Phase 1 に SeedVR2 worker、Phase 2 に SwiftVR と分かれるので
  組める。

inline の GPU 全体ピーク(同じ fork、RTX 5080、常駐約 1.85 GB 込み)は scale 2 で 480p
10.5 GB / 1080p 11.4 GB、scale 4 で 12.8 / 14.0 GB。12 GB 級では、scale 4 はオフライン、
scale 2 は inline を試して VRAM 逼迫(offloader の退避や worker の OOM 警告)が出るなら
オフライン、が実測に沿う。

### オフラインの挙動と制約

- 起動時の拒否は FlashVSR のオフラインと同じ: `--stream`、`--frame-gen`、
  `--retarget-high-fps`、`--segments`、VR モード(`--vr-mode auto` が VR を検出した場合を
  含む)、フォルダ入力、画像入力。
- `--max-clip-size` は利用者の指定(既定 90)のまま通る(FlashVSR オフラインの
  `--flashvsr-max-clip-frames` に相当する上限は無い)。`--swiftvr-scale`、
  `--swiftvr-view-window`、`--swiftvr-accel` は inline と同じ意味で効く。
- **bf16 は正規の経路**。FP8 が使えない GPU では driver が FP8 を外して bf16 で走り、inline の
  ような警告は出ない。
- **SeedVR2 一次と併用可**。inline の fp8-recon 自動有効化は無く、Phase 1 の一次は
  `--fp8-recon` を付けない限り標準の TRT 経路で走る(inline は同居の VRAM 予算のために自動で
  有効にする)。inline と同じ一次にしたければ `--fp8-recon` を明示する。
- encode は 2 回(Phase 1 の捨て出力と Phase 3 の最終)で、Phase 3 で source を再デコード
  する。この固定費は FlashVSR と同じで、SwiftVR は二次の処理時間が短いぶん割合が大きい
  (「[オフラインの実測](#オフラインの実測)」)。
- **段階再開**。`--swiftvr-bundle-dir` で bundle を永続化すると、失敗した走行を同じコマンドで
  再実行したとき Phase 2 は完了済みの clip を飛ばす(Phase 1 は再実行される)。Phase 2 で
  clip の途中に VRAM が尽きると driver は 1 回やり直し、再失敗なら非ゼロで終了して bundle を
  残す。
- **ディスク容量**。bundle は Phase 2 の非圧縮の復元クロップが支配し、1 フレーム
  3 × (256 × scale)² byte(scale 4 で 3 MiB、scale 2 で 0.75 MiB)。目安(scale 4 で
  モザイクを含む 1 分あたり約 8 GB)、`/tmp` が tmpfs の場合の注意、Phase 1 前の警告と
  Phase 2 前の容量検査は [FlashVSR のディスク容量](flashvsr.md#ディスク容量)と同じで、
  案内文のフラグ名が `--swiftvr-bundle-dir` になる。
- Phase 2 driver は Linux で `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定する
  (inline worker と同じ。fork e7f186b では FP8 のピークは 0.2 GB しか変わらないが、bf16 の
  12.4 GiB を 16 GB に収める余裕を残す)。Windows では `PYTHONUTF8=1`。
- frozen build では driver も `<dist>/jasna/restorer/` に実ファイルとして複製される。

### オフラインの実測

Linux、RTX 5080 16 GB、fork e7f186b。inline は同じ jasna と fork で同じ日に取り直した
(前掲の inline の表と誤差の範囲)。GPU 全体ピークはデスクトップ常駐(約 1.9 GB)込みの
`nvidia-smi` 0.5 秒ポーリング、段ごとのピークと時間はプロセス単位の値。既定の clip 90、
色補正 wavelet、view 平滑化 15。出力フレーム数は全走行で入力と一致した。

| 素材 | scale | inline 壁時計 / GPU 全体ピーク | オフライン壁時計 / GPU 全体ピーク | Phase 1 / 2 / 3 のプロセスピーク | Phase 1 / 2 / 3 の時間 |
|------|-------|------|------|------|------|
| 480p | 2 | 44.4 秒 / 10.3 GB | **62.8 秒 / 8.0 GB** | 2.9 / 6.4 / 0.5 GB | 27.3 / 27.9 / 2.1 秒 |
| 480p | 4 | 101.3 秒 / 12.6 GB | **118.0 秒 / 10.4 GB** | 2.9 / 8.8 / 0.5 GB | 27.3 / 80.7 / 4.7 秒 |
| 1080p | 2 | 54.7 秒 / 11.1 GB | **95.0 秒 / 8.0 GB** | 4.0 / 6.4 / 1.0 GB | 42.0 / 38.4 / 9.5 秒 |
| 1080p | 4 | 137.5 秒 / 13.8 GB | **172.9 秒 / 10.4 GB** | 4.0 / 8.8 / 1.0 GB | 41.5 / 113.9 / 12.2 秒 |
| 480p | 4、bf16(`--no-swiftvr-accel`) | (inline は OOM) | **186.6 秒 / 14.9 GB** | 2.9 / 13.3 / 0.5 GB | 27.4 / 149.7 / 4.3 秒 |

- GPU 全体ピークは Phase 2(SwiftVR 単体)で決まり、前掲の単体の表と一致する。Phase 1 は
  一次のみの走行と同じで、Phase 3 は軽い。
- 壁時計の差は Phase 1 の捨て encode と Phase 3 の再 decode と encode の固定費で、SwiftVR の
  処理時間が短いぶん inline との比は FlashVSR より大きい。
- **等価性ゲート**(inline 出力とオフライン出力を decode して `ffmpeg` の psnr フィルタで
  フレームごとに比較)。既定の encode(HEVC NVENC)では 480p で平均 46.22 dB
  (最小 42.16 dB)、1080p で 47.97〜48.85 dB
  (最小 42.07〜44.02 dB)だったが、この値は encode ノイズの
  床である。同じ素材で FP8 と bf16 のオフライン出力どうし(復元は本当に違う)も
  46.29 dB、FlashVSR の tiny と tiny-long どうしも 46.36 dB
  で区別がつかず、モザイクの無いフレームは inline とオフラインで bit 一致する
  (1490 / 4930 フレーム)。差を見るには encode を
  外す必要があるので、`--encoder-settings tune=lossless,spatial_aq=0,temporal-aq=0`(NVENC のロスレス。
  既定の適応量子化はロスレスと併用できないので外す)で 480p の scale 2 を取り直した:
  - inline 対オフライン(FP8): 平均 65.31 dB、最小 53.17 dB、完全一致 2221 / 4930 フレーム
  - オフラインの bf16 対 FP8(対照。復元が違うので差が出るはず): 平均 69.87 dB、最小 60.69 dB、完全一致 2221 / 4930 フレーム
  - inline 対オフラインの bf16: 平均 65.31 dB、最小 53.30 dB、完全一致 2221 / 4930 フレーム
  - inline 対オフライン(Phase 1 に `--fp8-recon` を明示): 全 4930 フレームが bit 一致(PSNR inf)
  inline との残差は一次復元の違いだった。inline は同居の VRAM 予算のために `--fp8-recon` を
  自動で有効にするが、Phase 1 は一次だけで走るので付けない。Phase 1 に `--fp8-recon` を明示
  すると全 4930 フレームが inline と bit 一致する。SwiftVR の段は同じクロップに対して
  決定的で(再開の試験でも bit 一致)、対照の bf16 は FP8 から 69.87 dB 離れて区別できる。
- **bf16**(`--no-swiftvr-accel`、scale 4、480p): driver は `accel: off` で bf16 のまま完走
  した。FP8 出力との PSNR は平均 46.29 dB(最小 42.12 dB)で、fork の
  FP8 対 bf16 の実測(約 47 dB)と同水準。
- **再開**: Phase 2 の途中(3 clip 完了時点)で driver を kill すると、走行は
  `SwiftVR run failed. Bundle kept for resume at:` と `--swiftvr-bundle-dir` の案内を出して
  非ゼロで終わった。同じコマンドで再実行すると Phase 2 は `done: 54 restored, 3 already present` で完走し、出力は
  中断なしの走行と全 4930 フレームが bit 一致(PSNR inf)。この機体は MPS + Exclusive_Process で、client を kill した直後は新しい CUDA
  コンテキストが `cudaErrorDevicesUnavailable` で拒否されるため、再実行は GPU が受け付ける
  まで待ってから行った(kill から 5 秒。jasna 側の制約ではない)。
- **SeedVR2 一次との併用**: 480p の 20 秒(500 フレーム)の素材で
  `--restoration-model-name seedvr2` + `swiftvr`(scale 2)が起動時に拒否されず完走
  (57.1 秒、GPU 全体ピーク 12.6 GB。Phase 1 は jasna 1.2 GB +
  SeedVR2 worker 9.8 GB、Phase 2 は 6.4 GB)。同じ素材で `swiftvr-inline` は起動時に `cannot be combined` で拒否された。
- **短い clip**: 480p の bundle は 57 clip で最短 2 フレーム(2 フレーム以下 4 個、view なし 0 個)、1080p は 58 clip で最短 2 フレーム(2 フレーム以下 1 個、view なし 0 個)。1 フレームの clip は view を持たず、Phase 3 は従来の経路で
  合成する(ユニットテストで担保)。
- **FlashVSR の回帰**: `flashvsr`(scale 2、`--flashvsr-accel`)のオフラインが version 2 の
  bundle(配置は全 clip で null: 57 / 57)で従来どおり完走した
  (144.0 秒、4930 フレーム。FlashVSR inline の旧出力との PSNR 平均
  46.36 dB は、tiny と tiny-long の違いによる)。

## SwiftVR distill(`--secondary-restoration swiftvr-distill`)

`swiftvr-distill` は、SwiftVR の出力から蒸留した小さな畳み込みネットワーク
(24 チャンネル、残差ブロック 12 個、約 13 万パラメータ)を SwiftVR 本体の代わりに
動かす。同じ 256px の一次復元クロップを 5 フレームずつ受け取り、中央フレームを
512px(2 倍)で返す。blend は `--swiftvr-scale 2` と同じく縮小合成で戻す。jasna の
プロセス内で PyTorch の FP32 で動くので、SwiftVR の checkout、venv、常駐 worker は
要らない。VRAM の上乗せは一次に対して約 0.2 GB、二次段の時間は scale 2 の
`swiftvr-inline` の 4 分の 1 から 6 分の 1 である。足されるのは輪郭の締まりと、復元
クロップから生成したもっともらしい質感であり、単体では復元モデルではなく、出力の
どこにも元の情報は含まれない。

モデル(`TinyROIEnhancer`、チェックポイント `roi-distill-pilot-v1`)は
[mioh](https://github.com/mioh-labs/mioh) の作者が、lada で復元したクロップに掛けた
SwiftVR を教師として蒸留し、無修正フレームに対する detail loss で追加学習したもの。
[`okatti/swiftvr-distill`](https://huggingface.co/okatti/swiftvr-distill) に
AGPL-3.0(jasna と同じ。リポジトリには `not-for-all-audiences` タグが付く)で公開
されている。jasna には同梱しない。`swiftvr-distill.pt` をそこから取得して、他のモデルと
同じ `model_weights/` に置く(別の場所に置くなら `--swiftvr-distill-model` で渡す)。
jasna が組むネットワークはこのリポジトリの README の定義どおりで、公開された
state dict はそのまま読み込める。

### 使い方

```bash
# 重みを model_weights/ へ(一度だけ)
wget -O model_weights/swiftvr-distill.pt \
  https://huggingface.co/okatti/swiftvr-distill/resolve/main/swiftvr-distill.pt

jasna --input in.mp4 --output out.mkv --secondary-restoration swiftvr-distill
```

`model_weights/` の探索順(`$JASNA_MODEL_WEIGHTS_DIR`、実行ファイルの隣、カレント
ディレクトリ、パッケージの隣)は他のモデルと同じ。重みが無ければ、エンジンコンパイルの
前に入手先を示して止まる。

### フラグ

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--swiftvr-distill-model` | `<model_weights>/swiftvr-distill.pt` | チェックポイントのパス。省略時は `model_weights/` の `swiftvr-distill.pt`、ファイル名だけなら `model_weights/` の中を探す。中身は `version`、`architecture`、`model` を持つ dict で、`weights_only=True` で読む。 |
| `--swiftvr-distill-view-window` | `15` | 切り出し view を前後 N フレームで平滑化する。`--swiftvr-view-window` と同じ機構(`0` で無効)。詳細は「[切り出し view の平滑化](#切り出し-view-の平滑化--swiftvr-view-window)」。 |
| `--swiftvr-distill-strength` | `0.75` | モデルが入力の双線形 2 倍に足す分の比率(0 から 2)。`0` は双線形拡大そのもの、`1` はモデルの出力。揺れも質感もおおむね比例する。 |
| `--swiftvr-distill-stabilize` | `0` | 実験的。足した分を、入力が近い範囲で前後 N フレーム(0 から 8)と混ぜる。jasna 独自の追加で、作者のアプリにはない処理。テストクリップでは強さを下げたのと同じ引き換えになり、実素材では未計測。 |

### 挙動と制約

- `rtx-super-res` と同じ枠の、同期のプロセス内 `SecondaryRestorer`。worker も
  ハンドシェイクも clip 上限も無い。中央フレーム 4 枚ずつのバッチで動くので、
  VRAM は clip 長に依存しない。
- 5 フレーム窓は clip の端のフレームを繰り返して作る。keep 範囲の外のフレームは
  返さないが、時間方向の文脈としては使う。
- `swiftvr-inline` と違い、起動時に強制も拒否もしない。fp8-recon は自動有効化
  されず、`--frame-gen` も切られず、SeedVR2 一次とも排他にならない(他に常駐する
  ものが無い)。色補正も無い。モデルは一次出力の双線形拡大に細部を足すだけなので、
  色は一次のままである。
- 作者のアプリは合成後のフレームを macOS の VideoToolbox の時間方向ノイズ
  フィルターにも通しており、安定性の多くをそれに負うとしている。Windows と Linux に
  同等物は無く、jasna では再現しない。ここで揺れを抑えているのは view の平滑化と
  既定の強さである。
- GUI には出ない。`--stream` と `--segments` は拒否しないが未検証(`--segments` は
  手元の素材ではスマートレンダリング自体が二次復元なしでも通らず、確かめられて
  いない)。VR は 8K SBS の素材 1 本を Linux で通し、完走して二次復元なしとフレーム数が
  一致し、利用者の目視でも効果を確認した。検証は Windows(RTX 5060 Ti)と
  Linux(RTX 5080)。

### 実測

Windows、RTX 5060 Ti 16 GB、一次は BasicVSR++(TensorRT)。1080p の実素材 1 本から、
モザイクの検出が 12 秒続く区間を 3 つ(各 361 フレーム)取った。一次クロップの動きが
小さい区間、中間の区間、大きい区間である。揺れの増加と質感は
mioh-labs の `swiftvr_view_smoothing.py`(揺れの報告の参考実装。「謝辞」を参照)の
`flicker_increase` / `texture_ratio` で、二次復元なしを基準とし、マスクは `swiftvr-inline` の出力から作った。`swiftvr-inline` は scale 2、
view window 15、高速化ありで動かした。3 区間の平均:

| 二次復元 | 揺れの増加 | 質感 | 二次段の時間 |
|---|---|---|---|
| `swiftvr-distill`、強さ 1.0、window 0 | +19.1% | 132.2% | 2.0 s |
| `swiftvr-distill`、強さ 1.0、window 15 | +13.9% | 133.7% | 1.9 s |
| `swiftvr-distill`、強さ 0.75、window 15(既定) | +8.6% | 122.4% | 2.0 s |
| `swiftvr-distill`、強さ 0.5、window 15 | +4.2% | 111.6% | 2.0 s |
| `rtx-super-res` 2 倍 | +0.8% | 106.1% | 0.8 s |
| `swiftvr-inline` scale 2 | +10.9% | 130.0% | 8.2 から 12.0 s |

GPU 全体のピークは、二次復元なしで約 6.0 GB、`swiftvr-distill` で 6.0 から 6.4 GB、
`swiftvr-inline` で 11.4 から 11.8 GB だった(常駐分約 2.7 GB を含む)。

- **view の平滑化はこのモデルにも効く。** 強さ 1.0 で揺れが +19.1% から +13.9% に
  下がり、質感は変わらない。動きの小さい区間では +38.7% が +25.7% になる
  (1080p のテストクリップでは 1 ポイントしか動かなかったが、それは切り出しの位置が
  ほとんど動かない素材だったためで、一般化できなかった)。
- **揺れと質感は強さに比例する。** 質感を揃えて比べると、揺れは scale 2 の
  `swiftvr-inline` より 1 から 2 ポイント大きい。
- **動きのある場面での働き方が違う。** 動きの大きい区間で `swiftvr-inline` は
  ほとんど何も足さない(質感 101.5%)が、`swiftvr-distill` は質感を 23% 上げる。
- **静止した入力では揺れない。** 同じフレームを並べた入力では足した分が厳密に
  一定になる。テストクリップでは、揺れの増加の約 7 割が質感を揃えた線形シャープでも
  出る量で、多くは輪郭を強めること自体に伴う。

既定値(強さ 0.75、window 15)はこの数値と、利用者が複数の素材を目視した結果で
決めた。揺れの違和感は少なく、質感も悪くない。輪郭は SwiftVR より締まり、面は
平滑になる。

### Linux での実測

Linux、RTX 5080 16 GB、2026-10-08。一次は BasicVSR++(TensorRT)、`swiftvr-inline` は
scale 2、view window 15、高速化あり。GPU 全体のピークはデスクトップ常駐(約 1.9 GB)を
含む `nvidia-smi` の 500 ms ポーリングの最大値。二次段の時間はログの
`[timing] secondary` の `restore`(同期復元器なので GPU の処理時間にほぼ等しい)で、
出力フレーム数は全走行で入力と一致した。

| 素材 | 二次復元 | 二次段 | 壁時計 | GPU 全体ピーク |
|---|---|---|---|---|
| テストクリップ 1080p、300 フレーム | なし | 0.0 s | 5.9 s | 4.8 GB |
| 同 | `swiftvr-distill`(既定) | 0.4 s | 5.3 s | 4.9 GB |
| 同 | `swiftvr-inline` scale 2 | 2.7 s | 35.3 s | 10.5 GB |
| 480p、4931 フレーム | なし | 0.0 s | 18.4 s | 4.5 GB |
| 同 | `swiftvr-distill`(既定) | 3.8 s | 21.7 s | 4.8 GB |
| 同 | `swiftvr-inline` scale 2 | 26.0 s | 42.3 s | 10.4 GB |
| 1080p、4203 フレーム | なし | 0.1 s | 24.0 s | 5.5 GB |
| 同 | `swiftvr-distill`(既定) | 5.8 s | 27.5 s | 5.7 GB |
| 同 | `swiftvr-inline` scale 2 | 38.9 s | 52.4 s | 11.2 GB |

- 二次段は scale 2 の `swiftvr-inline` の 6.7 から 6.8 分の 1。テストクリップは
  Windows(RTX 5060 Ti)の 1.0 s に対して 0.4 s。`swiftvr-inline` の壁時計には worker の
  起動(model load 9 s、warmup 9 s)が含まれる。
- GPU 全体ピークの上乗せは、二次復元なしに対して 0.05 から 0.3 GB。
- 強さ 0 の出力は、二次復元なしとロスレス encode 同士で Y 61 dB。一次クロップが
  双線形 2 倍と縮小合成を一往復するのでビット一致ではないが、目視で区別できる量では
  ない。`--swiftvr-distill-stabilize 2` はテストクリップで二次段 +0.1 s、VRAM +0.44 GB。
- 8K SBS の VR 素材(2998 フレーム)では、二次復元なしが 103.6 s / 14.0 GB、
  `swiftvr-distill` が 112.1 s / 13.8 GB で、ピークの差は測定誤差の範囲(一次段が
  支配的)。

揺れと質感の指標は、Windows と同じ `flicker_increase` / `texture_ratio` を、上の 1080p
素材の出力から取った 361 フレームの区間 3 つ(マスク面積の大きい順に選んだ)で求めた。
基準は二次復元なし、マスクは `swiftvr-inline` の出力から作った。3 区間の平均:

| 二次復元 | 揺れの増加 | 質感 |
|---|---|---|
| `swiftvr-distill`、強さ 1.0、window 0 | +3.4% | 139.3% |
| `swiftvr-distill`、強さ 0.75、window 15(既定) | +2.0% | 119.9% |
| `swiftvr-inline` scale 2 | +1.0% | 212.4% |

質感の順位(inline が最大、強さ 1.0 がその次、既定が最小)と、既定が強さ 1.0 / window 0
より揺れないことは Windows と同じである。一方、`swiftvr-inline` の揺れは既定の
`swiftvr-distill` を下回り、Windows の順位(+10.9% 対 +8.6%)と逆になった。この素材は
3 区間とも動きが大きく(基準の輝度のフレーム間差が 9 から 10 階調)、揺れの増加が
どの構成でも Windows の 3 分の 1 以下に収まっているので、順位は素材で変わる。
数値そのものは素材に依存するため、Windows の表とは比べない。

利用者が 1080p と 480p の出力を二次復元なし、`swiftvr-inline` scale 2 と見比べ、8K SBS の
VR 出力も確かめて、問題なし(8K SBS でも効果が見える)と判断した。

### 実装

`jasna/restorer/swiftvr_distill_model.py`(ネットワークとチェックポイントの検査:
既知の version、`architecture` の範囲、有限値、strict な読み込み)、
`swiftvr_distill_secondary_restorer.py`(復元器、強さ、安定化)、
`session_config.py` / `session_factory.py` / `main.py` の配線(モデルのパスは他の重みと
同じ `model_weights/` の探索で決め、エンジンコンパイルの前にも検査する)。テストは
`tests/test_swiftvr_distill.py`(CPU。実重みのケースは `model_weights/swiftvr-distill.pt` が
あるか、`JASNA_SWIFTVR_DISTILL_MODEL` がチェックポイントを指すときに走る)と
`tests/test_main.py`。

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

- `jasna/restorer/swiftvr_common.py`: `--swiftvr-*` の登録、パスの解決、オフライン Phase 2
  のコマンド(torch を import しない)。
- `jasna/restorer/swiftvr_inline_secondary_restorer.py`: 同期 `SecondaryRestorer`。
  worker の起動とハンドシェイク、wire の入出力、RGB と BGR の反転、keep window の切り出し。
  view 平滑化の窓幅を `view_smoothing_window` 属性で申告する。FlashVSR inline restorer
  と同じ構造で、パッチ検査、高速化の環境変数、実行中の降格報告と respawn を持たない。
- `jasna/restorer/swiftvr_inline_worker.py`: SwiftVR venv で動く worker。jasna も
  lada も import しない(lada-ex にそのまま持ち込める)。高速化の判定、モデル読込、
  warmup、clip ごとの `restore_clip()` と色補正。色補正の primitive(wavelet / AdaIN)は
  FlashVSR worker と同じ数式を内蔵する(両者が bit 一致することをテストで確認)。
  lada-ex 側の複製と同一ファイルを保つため、path 読み込みでの共有はしない。
- `jasna/restorer/swiftvr_phase2_driver.py`: オフライン Phase 2 の driver(SwiftVR venv)。
  jasna を import せず、隣の worker を path で読んで高速化の判定、モデル読込、warmup、
  `restore_clip()` の枚数検査付き呼び出し、GPU 上の色補正を共有する。bundle の clip を順に
  読み、完了済みは飛ばす。
- `jasna/restorer/flashvsr_offline.py`: FlashVSR と共用のオフライン orchestrator。engine
  レコード(`OfflineEngine`)でパス解決とフラグ名、scale、Phase 1 の clip 上限(FlashVSR
  のみ)と view の窓幅(SwiftVR のみ)、Phase 2 のコマンド(`swiftvr_common.swiftvr_phase2_command()`)
  を差し替える。Phase 1 の dump hook は `RestorationPipeline.view_smoothing_window` を
  設定値に差し替えて view を作り、bundle version 2 の `view_placements` として書く。Phase 3
  は配置があれば view の経路で blend する。
- `jasna/tracking/crop_view.py`: 切り出し view の幾何(配置の計算、平滑化、
  クランプ、view への再サンプル、blend のサンプル)。`restorer/restoration_pipeline.py`
  が二次 restorer の `view_smoothing_window` を見て一次段の末尾で view を作り、
  `pipeline_items.py` の `view_placements` で配置を運び、`blend_buffer.py` が配置の
  あるときだけ view から直接合成する(0 または属性なしで従来経路。FlashVSR inline にも
  同じ機構を足せる)。
- `jasna/session_config.py` / `session_factory.py` / `main.py`: 設定フィールド、
  restorer の生成、起動時検査(fp8-recon 自動有効化、frame-gen と SeedVR2 一次の拒否)。
- `scripts/build_nuitka.py`: worker と Phase 2 driver を実ファイルとして
  `<dist>/jasna/restorer/` に複製する(FlashVSR worker と並べて置く。driver は worker を
  path で読む)。
- テスト: `tests/test_swiftvr_inline.py`(stub worker で wire、フラグ、ハンドシェイク、
  色補正の GPU 版と FlashVSR 版の一致)、`tests/test_swiftvr_offline.py`(両 engine の
  起動時検査、Phase 2 のコマンド、bundle version 2 の往復と version 検査、Phase 1 hook の
  view、Phase 3 の view 合成が inline 経路と一致すること、driver)、`tests/test_main.py`(choices と既定値)、
  `tests/test_crop_view.py`(view の幾何: 配置が元の格子なら従来経路と一致、平滑化した
  view の往復、クランプの被覆)、`test_restoration_pipeline.py` と `test_blend_buffer.py`
  の view 経路。

SwiftVR fork 側: `swiftvr/runner.py` の `restore_chunk()`(オフライン runner と共有)、
`swiftvr/pipeline.py` の `restore_clip()`。

## 謝辞

scale 2 の時間安定性の改善(切り出し view の平滑化)は、
[mioh-labs](https://github.com/mioh-labs/mioh) の報告と参考実装による。揺れの原因が
SwiftVR そのものではなく SwiftVR に渡す切り出しの見え方がフレームごとに動くことに
あると突き止め、切り出しの位置と倍率を前後 15 フレームの移動平均で平滑化して出力を
元の枠に戻す対策を、プローブ実験と評価指標とともに公開してくれた。jasna の実装は
その幾何をパイプラインの一次段と blend 段に組み込んだもので、鏡余白とクランプは
jasna 側の設計である。

SwiftVR distill のモデル(`roi-distill-pilot-v1`)は、同じ mioh の作者 okatti 氏が
SwiftVR を教師として蒸留し、学習済みの重みとネットワークの定義を
[`okatti/swiftvr-distill`](https://huggingface.co/okatti/swiftvr-distill) に AGPL-3.0 で公開してくれた
ものである。jasna への組み込みは、同氏が提案書(本リポジトリの PR #4)で示した構成に
沿って行った。復元器の枠組み(同期の `SecondaryRestorer`、view 平滑化の再利用)、強さと
安定化のオプション、既定値の選定は jasna 側の設計である。

## Windows での注意事項

inline(`swiftvr-inline`)は Windows 11、RTX 5060 Ti 16 GB で検証した(2026-09-27、
jasna `32168df`、SwiftVR fork `e7f186b`)。オフライン 3 段(`swiftvr`)も同じ機械で検証した
(同日、jasna `afe8d18`。「[オフライン 3 段の実測](#オフライン-3-段の実測windows)」)。

- セットアップは Linux と同じ。Triton は `triton-windows` が `uv sync` で入り、
  C++ コンパイラも開発ヘッダも要らない。高速化(FP8 DiT、torch.compile)は判定を通って
  有効になった。`--swiftvr-python` の既定は `<repo>/.venv/Scripts/python.exe`。
- worker の起動は model load 5〜6 秒、warmup 11〜12 秒(チェックポイントがページ
  キャッシュに乗った状態)。初回だけ、Triton の事前検査カーネルのコンパイルで
  `remark: ... instructions in function` という 1 行が worker の stderr から出る。無害で、
  カーネルキャッシュが温まった後は出ない。
- **16 GB に収まる**。`expandable_segments` が使えなくても、GPU 全体のピークは
  1080p scale 4 で 13.2 GB(デスクトップ常駐 1.5 GB を含む。搭載 15.9 GB)で、天井まで
  2.7 GB 残る。常駐が多い環境ではその分だけ上がる(常駐 2.5 GB の走行では 14.2 GB)。
- **`--no-swiftvr-accel`(bf16)は OOM で止まらず完走した**(480p、scale 4)。Linux と
  異なる。壁時計は高速化ありの約 1.9 倍、GPU 全体のピークは 15.6 GB で搭載量(15.9 GB)の
  上限に張り付いた。Windows ドライバの CUDA Sysmem Fallback(既定で有効)が不足分を
  システム RAM へ逃がしたか、480p では一次の VRAM が小さく辛うじて収まったかは
  切り分けていない。Sysmem Fallback を無効にした環境や大きい素材では OOM になり得るので、
  16 GB カードで高速化が使えない場合はオフラインの `swiftvr` を使う。
- 検証用の環境変数 `JASNA_SWIFTVR_COLOR_FIX` は、PowerShell では
  `$env:JASNA_SWIFTVR_COLOR_FIX="none"` で設定し、走行後に
  `Remove-Item Env:JASNA_SWIFTVR_COLOR_FIX` で消す(シェルに残すと以後の走行すべてに効く)。
  Git Bash なら `JASNA_SWIFTVR_COLOR_FIX=none jasna ...` とコマンド単位で渡せる。
- view 平滑化は jasna 側の GPU 処理だけなので、Windows 固有の要素は無い。コストは
  Linux と同じく誤差の範囲だった。

実測(Windows 11、RTX 5060 Ti 16 GB。素材は Linux と異なり 480p が 10661 フレーム、
1080p が 6242 フレームなので、壁時計は上の Linux の表と比較しない。GPU 全体のピークは
`nvidia-smi` の 0.5 秒ポーリングの最大値で、デスクトップ常駐 1.4〜1.6 GB を含む。
GB は MiB / 1024。出力フレーム数は全走行で入力と一致した):

| 素材 | 構成 | 壁時計 | GPU 全体ピーク |
|------|------|--------|----------------|
| 480p | 一次のみ | 114 秒 | 3.7 GB |
| 480p | `swiftvr-inline` scale 4 | 671 秒 | 12.8 GB |
| 480p | `swiftvr-inline` scale 2 | 280 秒 | 9.8 GB |
| 480p | `swiftvr-inline` scale 4、色補正 none | 613 秒 | 13.0 GB |
| 480p | `swiftvr-inline` scale 4、`--no-swiftvr-accel` | 1287 秒 | 15.6 GB |
| 1080p | `swiftvr-inline` scale 4 | 477 秒 | 13.2 GB |
| 1080p | `swiftvr-inline` scale 2 | 189 秒 | 10.4 GB |
| 1080p | `swiftvr-inline` scale 2、`--swiftvr-view-window 0` | 197 秒 | 10.4 GB |
| 1080p | `flashvsr-inline` scale 2(以前の実測) | 463 秒 | 11.0 GB |
| 1080p | `flashvsr-inline` scale 4 tiles 1(以前の実測) | 1555 秒 | 15.5 GB |

- 同機の FlashVSR inline に対して、1080p で scale 2 は約 2.4 倍、scale 4 は約 3.3 倍速い。
- 色補正のコストは壁時計の約 9.5%(671 秒 対 613 秒)。Linux の 4.6% より大きく、
  FlashVSR の Windows 実測(約 10%)と同程度。
- `--no-swiftvr-accel` の行はデスクトップ常駐 2.9 GB、FlashVSR の行は scale 2 が 3.1 GB、
  scale 4 が 1.3 GB の状態で測った。
- 色ずれ(480p、一次のみ出力比。「補正なし」出力が変えた画素を共通のマスクにして
  10 フレームごとに測った。Linux のゲートとマスクの定義が違うので絶対値は比べない):
  チャネル毎 median |Δmean| は補正なし 3.98 → wavelet scale 4 0.71、scale 2 0.66
  (合格。94% 以上のフレームで wavelet が補正なしを下回る)。
- 目視(利用者): scale 4、scale 2(view 平滑化の有無)の出力を FlashVSR inline の
  同素材出力と並べて確認し、問題は無かった。

### オフライン 3 段の実測(Windows)

同じ機械と素材で、オフライン 3 段の全項目が完走し、出力フレーム数は入力と一致した。

- Phase 2 driver は `<repo>\.venv\Scripts\python.exe` で起動され(`PYTHONUTF8=1`)、
  高速化は判定を通って有効になった。ready は model load 6〜18 秒、warmup 8〜24 秒。
- `nvidia-smi` のプロセス別の VRAM は WDDM では取れないので、GPU 全体の値を段の区間で
  切り分けて、走行開始前の常駐を引いた増分を各段のピークとした。
- **Phase 2 のピークは Linux の代理測定(`expandable_segments` なし)と 0.2 GB 以内**:
  FP8 の scale 4 で 9.2 GB、scale 2 で 6.3 GB(480p と 1080p で同じ)。bf16 の scale 4 は
  14.0 GB。
- GPU 全体の増分は inline より 2.1〜2.5 GB 小さく、壁時計は inline の 1.4〜1.7 倍。
- ロスレス encode で Phase 1 に `--fp8-recon` を明示すると、inline と全 10661 フレームが
  bit 一致した(Linux と同じ)。`tune=lossless` はエラーなく通る。
- Phase 2 の途中で driver を止めると bundle が残って再開の案内が出て、同じコマンドの
  再実行で済んだ clip を飛ばして完走し、出力は中断なしと bit 一致した。直後の再実行でも
  Linux で見た `cudaErrorDevicesUnavailable` は出なかった。
- FlashVSR のオフライン(`flashvsr`)も従来どおり完走した(回帰なし)。

| 素材 | 構成 | 壁時計 | 常駐 | GPU 全体ピーク | Phase 1 / 2 / 3 増分 | Phase 1 / 2 / 3 時間 | bundle |
|------|------|--------|------|----------------|----------------------|----------------------|--------|
| 480p | scale 4 | 913 秒 | 2.0 GB | 11.2 GB | 2.7 / 9.2 / 0.2 GB | 215 / 633 / 59 秒 | 30 GB |
| 480p | scale 2 | 484 秒 | 1.9 GB | 8.2 GB | 2.8 / 6.3 / 0.3 GB | 224 / 224 / 33 秒 | 8.0 GB |
| 480p | scale 4、`--no-swiftvr-accel` | 1392 秒 | 1.4 GB | 15.4 GB | 2.7 / 14.0 / 0 GB | 213 / 1120 / 54 秒 | — |
| 1080p | scale 4 | 691 秒 | 0.4 GB | 9.6 GB | 3.2 / 9.2 / 0.7 GB | 159 / 490 / 38 秒 | 22 GB |
| 1080p | scale 2 | 323 秒 | 0.4 GB | 6.7 GB | 3.2 / 6.3 / 0.7 GB | 150 / 147 / 22 秒 | 5.9 GB |
| 480p | `flashvsr` scale 2、`--flashvsr-accel` | 962 秒 | 0.4 GB | 6.4 GB | 2.7 / 6.0 / 0.3 GB | 224 / 702 / 32 秒 | — |

常駐は検証中に 2.0 GB から 0.4 GB まで下がったので、比べるときは増分を使う。
12 GB 級(RTX 4070、5070)で scale 4 の FP8 を走らせる場合、常駐 2 GB 込みで約 11.2 GB
(480p scale 4 の行がちょうどこの条件)になる。
