# 自前 LTX 復元モデルの学習（`+modi`）

このフォークは公式の LTX 重みを復号できませんが、jasna の LTX 推論経路は公開コードで、
モデルフォルダの平文 safetensors を読み込みます。本書はその経路向けのモデルを自分で作る
手順です。公開の LTX-2.3 チェックポイントを jasna のファイル形式に変換し、モザイク/原画の
window ペアで IC-LoRA（参照動画で条件付けする LoRA）を学習して復元能力を持たせます。
スクリプトは [`scripts/ltx_lora_training/`](../../scripts/ltx_lora_training/README.md)、
データセットの dump は作者の lada-ex fork 側にあります。

期待値を先に書きます。素のチェックポイントは変換して動きますが、何も復元しません。参照と
して渡したモザイク crop をそのまま写すだけです。本書の処方で学習した IC-LoRA を融合すると
モザイクを復元しますが、作者の確認では公式リリースのモデルより目に見えて柔らかい仕上がり
です。protection モジュールのないビルドで動く LTX モデルを得る手段であって、リリース版の
重みの代替ではありません。学習済みモデルそのものは公開せず、道具と処方を公開します。

本書の内容はすべて Linux x86-64（RTX 5080、16 GB）で実行したもので、Windows と ARM64 は
未検証です（[既知の制限](#既知の制限)）。

## 必要なもの

| | |
| --- | --- |
| GPU | VRAM 16 GB。`train_ic_lora.sh` の設定で学習のピークは 12.6 GB。テキストエンコードの工程は Gemma 3 12B の一部を VRAM（`--gpu-budget`、既定 11 GiB）に、残りをホスト RAM に置く。 |
| システム RAM | 64 GB 推奨。学習で 32 GB 使用。トレーナーは 48 ブロック中 36 をホストメモリに退避する。 |
| ディスク | Sulphur-2 チェックポイント約 43 GB、Gemma 3 12B 約 23 GB、LTX-2.5 VAE 1.5 GB、学習ペア 1300 本で 4.6 GB、キャッシュ 3 GB、変換後モデル 1 本 13.4 GB（bf16 は 26.3 GB）。 |
| jasna | このフォークの開発インストール（[Linux](building_linux.md)、[Windows](building_windows.md)）。変換器、`encode_prompt.py`（jasna の依存に含まれる `transformers` を使う）、window 確認はその venv で動く。 |
| トレーナー | [Musubi Tuner の LTX-2 fork](https://github.com/AkaneTendo25/musubi-tuner)（ブランチ `ltx-2-dev`、コミット `9dc9abc` で使用）。専用 venv にチェックアウト内で `uv sync --extra cu130 --python 3.12`。 |
| データセットの道具 | [lada-ex](https://codeberg.org/sh202603/lada-ex) のチェックアウト（ブランチ `feat/seedvr2-lora-primary`）とその venv、および `mosaic_removal_vid` データセット（lada-ex が BasicVSR++ の学習に使う原画 crop 群。作り方は lada-ex の `docs/training_and_dataset_creation.md`）。 |
| 取得するモデル | [SulphurAI/Sulphur-2-base](https://huggingface.co/SulphurAI/Sulphur-2-base) の `sulphur_distil_bf16.safetensors`、[Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) の `vae/ltx-2.5-video-vae-bf16.safetensors`（gated。ライセンスに同意して取得）、[google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)（gated、Gemma 利用規約）。[ライセンス](#ライセンス)を参照。 |

## 全体の流れ

```
lada-ex mosaic_removal_vid ──(1) ltx-ic-lora-dump-pairs.py──> gt/lq の 121 フレーム window ペア、512 px、固定キャプション
                                                                      │
キャプション ──(2) encode_prompt.py（Gemma 3 + LTX-2.3 の projection/connector）──> prompt_context.safetensors
                                                                      │               │
                               (3) cache_latents.sh + write_te_cache.py <──────────────┘
                                                                      │
                               (4) train_ic_lora.sh（Musubi Tuner、IC-LoRA v2v、rank 32）──> LoRA checkpoint
                                                                      │
LTX-2.3 チェックポイント ──(5) convert_ltx_to_jasna.py --lora ──> transformer ファイル ──(5) make_bundle.py + LTX-2.5 VAE ──> モデルフォルダ
                                                                      │
                               (6) val ペアで ltx_window_infer.py、jasna --restoration-model-name ltx-8step --restoration-model-path <フォルダ>
```

jasna のコードに手を入れずに済む理由。Sulphur-2-base は LTX-2.3 の構造改変なしのフル
ファインチューンで、LTX-2.3 と LTX-2.5 のビデオ側ブロックはテンソル名と形状が一致します
（2.3 には FFN バイアスがあり、jasna はバイアスを任意扱いにする。2.5 にはキーフレーム埋め込み
があり、変換器がゼロで追加する）。2.5 の VAE エンコーダは 2.3 と同一なので、2.3 系で学習した
モデルの latent は jasna の 2.5 diffusion decoder でそのまま復号できます。jasna は各 window を、
目標トークンの後ろに参照トークン（VAE でエンコードしたモザイク crop）を同じ位置で連結し、
参照側の σ を 0 にして条件付けします。これは公開トレーナーの IC-LoRA の参照レイアウトと
同じです。詳細と処方の根拠となる計測は作者の設計メモにあります（本リポジトリには含まない）。

## 手順 1: データセット

ペアは lada-ex の `mosaic_removal_vid` データセットから作ります。原画クリップに lada 自身の
劣化コードで合成モザイクを置いたものです。dump スクリプトはそれを jasna の学習幾何に切り、
学習が見るものと推論が見るものを一致させます。

- クリップを「フレーム全体」とみなし、モザイクのブロックマスク（jasna が検出を膨張するの
  と同じ 20 px 膨張）に jasna 自身の `track_camera` を当て、hop 88 の 121 フレーム window
  を取り、各 window を `crop_to_canvas` で 512 px の canvas に切る。
- `lq`（参照）は canvas 上で lada の劣化処方を掛けたモザイク、`gt`（目標）は原画の crop、
  `mask` は canvas 上のモザイクブロックマスク。
- クリップの実領域外は jasna の letterbox と同じ流儀で鏡像埋めする。埋めが 35 % を超える
  window は落とす（`--max-outside`）。
- 各 `gt` クリップの隣に固定キャプションを書く（`--caption`。既定は作者が使った文）。手順 2
  でエンコードする文と同じでなければならない。

```bash
cd <lada-ex checkout>
JASNA_ROOT=<jasna checkout> .venv/bin/python scripts/dataset_creation/ltx-ic-lora-dump-pairs.py \
    --meta-dir datasets/mosaic_removal_vid/train/crop_unscaled_meta --num-clips 0 \
    --min-scene-side 512 --output-dir datasets/ltx_ic_lora_pairs/train
JASNA_ROOT=<jasna checkout> .venv/bin/python scripts/dataset_creation/ltx-ic-lora-dump-pairs.py \
    --meta-dir datasets/mosaic_removal_vid/val/crop_unscaled_meta --num-clips 0 \
    --min-scene-side 512 --max-windows-per-clip 1 --output-dir datasets/ltx_ic_lora_pairs/val
```

各実行は `gt/`、`lq/`、`mask/`、`params/`（ペアごとの幾何と劣化の記録）、`dataset.json`、
`manifest.json`、`musubi_dataset.toml` の雛形を書きます。作者のデータセットでは学習ペア約
1300 本（8 worker、1 ペア約 10 秒）と検証ペア 115 本になりました。トレーナー用に小さな検証
サブセットを作ります。検証ペア二十数本の `gt/` と `lq/` を `val24/` にコピーしてください
（トレーナーは 250 step ごとにこれで検証を回す。24 本で約 2.5 分）。

## 手順 2: キャプションを `prompt_context` にする

jasna にはテキストエンコーダがありません。固定キャプションを ltx-core が LTX-2.3 で行うとおり
（Gemma 3 12B の hidden state、チェックポイント自身の `text_embedding_projection` と
`video_embeddings_connector`）に一度だけエンコードし、結果を `prompt_context` としてモデル
ファイルに埋め込みます。学習も同じテンソルをテキストキャッシュとして使う（手順 3）ので、
学習と推論は同じ定数で条件付けされます。

```bash
cd <jasna checkout>
.venv/bin/python scripts/ltx_lora_training/encode_prompt.py \
    --gemma-dir <gemma-3-12b-it> --source <sulphur_distil_bf16.safetensors> \
    --prompt "$(cat <lada-ex>/datasets/ltx_ic_lora_pairs/train/gt/<任意のペア>.txt)" \
    --output prompt_context.safetensors --hidden-cache gemma_hidden.safetensors
```

Gemma は VRAM を `--gpu-budget`（既定 11 GiB）だけ使い、23 GB の残りはホスト RAM に置きます。
`--hidden-cache` は積み上げた hidden state を保存し、Gemma を読み直さずに projection と
connector だけをやり直せるようにします。

## 手順 3: latent キャッシュとテキストキャッシュ

`scripts/ltx_lora_training/musubi/dataset.toml.example` をコピーし、`/path/to/...` の項目
（`train/gt`、`train/lq`、`val24/gt`、`val24/lq` の各フォルダと 4 つのキャッシュフォルダ）を
置き換えてから、次を実行します。

```bash
export MUSUBI_DIR=<musubi-tuner checkout> LTX_CHECKPOINT=<sulphur_distil_bf16.safetensors> DATASET_TOML=<your dataset.toml>
scripts/ltx_lora_training/musubi/cache_latents.sh
.venv/bin/python scripts/ltx_lora_training/musubi/write_te_cache.py --prompt-context prompt_context.safetensors \
    <cache>/train <cache>/val
```

`cache_latents.sh` は目標と参照のクリップをチェックポイントの VAE でエンコードします
（1300 ペアと参照で RTX 5080 では 52 分、キャッシュ約 3 GB）。`write_te_cache.py` は、各 latent
に必要なテキストキャッシュを Musubi の Gemma 工程の代わりに `prompt_context` から書きます。
Musubi の Gemma 工程は 8-bit でも 16 GB に収まりません。対象は目標側のキャッシュフォルダ
（`train`、`val`）だけで、`_ref` フォルダには掛けません。

## 手順 4: 学習

```bash
export OUT_DIR=<実行結果のフォルダ>
scripts/ltx_lora_training/musubi/train_ic_lora.sh <run-name> 2500
```

処方（全フラグはスクリプトの冒頭コメント）: IC-LoRA の `v2v` プリセット、rank 32、学習率
1e-4 固定、adamw8bit、bf16 混合精度と fp8 のベース重み、36 ブロックをホストメモリへ退避
（ホストからデバイスへの転送のみ）、bounded activation offload、データローダ worker 2、
500 step ごとに checkpoint、250 step ごとに検証。RTX 5080 では 11〜12 s/step、VRAM 12.6 GB、
RAM 32 GB で、2500 step に約 8.5 時間かかりました。退避ブロックを減らすと backward で VRAM
が足りなくなり、worker 8 では RAM を 58 GB 使います。

checkpoint の選び方について、作者の実行から分かったこと。

- 学習率固定では約 2600 step を過ぎて発散しました（学習損失が跳ね、検証損失も上昇）。2500
  で止めるか、こまめに保存して最終ではなく下の window 確認で checkpoint を選んでください。
- checkpoint から学習率を下げて再開する、減衰スケジュールにする、低ノイズ側へ timestep を
  寄せる、モザイク周辺の軟らかい損失マスクを使う、のいずれも出力を滑らかにはしても鮮明には
  しませんでした。既定ではなく実験として扱ってください。
- 検証損失は学習の向きは示しますが見た目の鮮鋭さは示しません。checkpoint は復号した動画で
  判断してください。

checkpoint は `$OUT_DIR/<run-name>/<run-name>-stepNNNNNNNN.safetensors` に、最終は
`<run-name>.safetensors` に書かれます。`--resume` で再開すると保存済みの optimizer 状態を
引き継ぎ、コマンドラインで変えた学習率は無視されます。

## 手順 5: 変換とモデルフォルダの作成

```bash
cd <jasna checkout>
.venv/bin/python scripts/ltx_lora_training/convert_ltx_to_jasna.py \
    --source <sulphur_distil_bf16.safetensors> --prompt-context prompt_context.safetensors \
    --lora $OUT_DIR/<run-name>/<run-name>-step00002500.safetensors --output my-ltx-int8.safetensors
.venv/bin/python scripts/ltx_lora_training/make_bundle.py --transformer my-ltx-int8.safetensors \
    --vae <ltx-2.5-video-vae-bf16.safetensors> --model-dir <モデルフォルダ>
```

変換器はビデオ側のテンソルを改名し、音声・テキスト・VAE の枝を除き、LoRA を fp32 で重みに
融合し、ゼロのキーフレーム埋め込みと `prompt_context` を加え、distilled 8-step の σ 列を
メタデータに書き、各ブロックの線形層を ConvRot W8A8（リリース版の重みと同じ形式。
`--quant bf16` なら 2 倍のサイズで bf16 のまま）に量子化します。43 GB のチェックポイントの変換
は RTX 5080 で 1 分弱で、出力はストリーミングで書くので RAM は制約になりません。
`make_bundle.py` は VAE を jasna が記録しているテンソル表と照合し、8-step スロットの
ファイル名でリンクします。v0.12.1 ベースから decoder のファイル名は
`vae-decoder-full.safetensors` です（以前は `vae-decoder.safetensors`）。以前に作ったフォルダには
この名前を足してください（フォルダ内で `ln vae-decoder.safetensors vae-decoder-full.safetensors` など）。

このフォルダで jasna を実行します。

```bash
jasna --input in.mp4 --output out.mp4 --restoration-model-name ltx-8step --restoration-model-path <モデルフォルダ>
```

3 つのファイルを `model_weights/ltx-restore/` にコピーすると、開発インストールでは
`--restoration-model-name ltx-8step`（と GUI の 8-step LTX）の既定になります。
`--secondary-restoration rtx-super-res` は実行中に LTX の window を鮮明化します
（[changes_vs_upstream.md §3.7](changes_vs_upstream.md#37-ltx-実行中の-window-の鮮明化--secondary-restoration-rtx-super-res)）。
低解像度の素材では、この処方のモデルの柔らかさをある程度取り戻します。

## 手順 6: checkpoint の確認

```bash
.venv/bin/python scripts/ltx_lora_training/ltx_window_infer.py --model-dir <モデルフォルダ> \
    --out-dir <out> --inputs <lada-ex>/datasets/ltx_ic_lora_pairs/val/lq/<ペア>.mp4 ...
```

このスクリプトは `lq` クリップ 1 本を学習時と同じく 1 window として与え（検出も camera も
なし）、復号した window を書き出します。ペアの `gt` クリップや他の checkpoint の出力と比べて
ください。モデルの読み込み後は 1 window 約 7 秒です。合成クリップで full pipeline を回して
checkpoint を判断しないでください。jasna の検出器は合成モザイクを複数の track に分けるので、
window が学習幾何と合わなくなります。最終確認は実素材を pipeline に通して行います。

## 既知の制限

- **検証は Linux x86-64 のみ。** スクリプトは素の Python と bash で、Windows と ARM64 は
  未検証です。jasna の INT8 LTX 経路はブロックの線形層を自前の Triton カーネルで回すので、
  それが使えない環境では bf16 モデル（`--quant bf16`、26 GB）が要ります。作者の 16 GB カードでは
  bf16 モデルが jasna の torch.compile 経路で、大半のブロックをホスト RAM からストリーミング
  して動き、window あたり INT8 モデルの約 2 倍の時間で、出力は INT8 モデルとほぼ一致しました。
  この経路が Windows on ARM でビルドできるかは不明です。
- **土台は distilled 8-step モデル。** LoRA は Sulphur の distilled チェックポイントで学習し、
  jasna の 8-step スロットで動かしています。dev チェックポイントで `ltx-undistilled` スロット
  （15 step、STG あり）向けに学習することは未試行です。
- **native LTX ブロックには対応する CUDA toolkit が要る。** jasna が native ブロックのビルド
  失敗をログに出したときは torch.compile 経路に落ちて遅くなります。`CUDA_HOME` を GPU に対応
  した toolkit に向けてください。
- **品質はリリース版モデルに届かない。** 鮮明さが増したのは decoder 側の作業と
  `--secondary-restoration rtx-super-res` で、手順 4 に挙げた学習のノブではありませんでした。

## ライセンス

ここのスクリプトはこのフォークのコード（AGPL-3.0）で、重みは含みません。取得するモデルには
それぞれの条件があり、ダウンロード時に受け入れることになります。

- **Sulphur-2-base** は [LTX-2 Community License Agreement](https://huggingface.co/SulphurAI/Sulphur-2-base/blob/main/LICENSE.txt)
  をそのまま同梱しています。Lightricks LTX-2.3 の派生物で、そのライセンスが派生物に同じ
  ライセンスを付けることを求めるためです。
- **LTX-2.5 VAE** は [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)（Hugging Face で gated）。
- **Gemma 3 12B** は [Gemma Terms of Use](https://ai.google.dev/gemma/terms)（gated）。

この処方に関わる LTX ライセンスの要点を、作者の読み（法的助言ではない）として書きます。
利用、微調整、派生物の作成は使用制限の範囲で目的を問わず許可されます。年商 1,000 万ドル以上
の事業体は有償の商用ライセンスが必要です。学習したモデルは「Derivative of LTX-2」なので、
配布する場合はライセンス全文を添え、使用制限を受領者に伝える義務があります。変換器は出力の
メタデータに元ファイル名とライセンス名を記録します。
