# kukkiri の LTX 重みを使う（`+modi`）

[kukkiri](https://github.com/treehog2/kukkiri) は、モザイク復元用に fine-tune した LTX-2.5 の
重み `kiyo-ltx-v1` を平文の safetensors で公開しています。
このフォークは公式リリースの LTX 重み（暗号化済み）を読めませんが、jasna の LTX 推論経路は
公開コードなので、この重みを jasna の形式に変換すれば `--restoration-model-name ltx` で動きます。
本書はその手順です。

重みそのものは jasna に含まれません。
LTX-2.x Community License の派生物なので、配布元から各自で取得してください
（[ライセンスと配布](#ライセンスと配布)）。

本書の内容は Linux x86-64（RTX 5080、16 GB）と Windows（RTX 5060 Ti、16 GB）で実行したものです。

## 必要なもの

| | |
| --- | --- |
| GPU | INT8 の LTX 経路が動く NVIDIA GPU。16 GB で確認。 |
| ディスク | 取得する重み 14.8 GB と、変換後の transformer 13.4 GB。VAE はハードリンクなので増えません。 |
| jasna | このフォークの開発インストール（[Linux](building_linux.md)、[Windows](building_windows.md)）。ビルド済みの配布物（frozen build）では使えません（[既知の制限](#既知の制限)）。 |

## 1. 重みを取得する

配布元は <https://huggingface.co/treehog/kukkiri> の `ltx-v1/` で、認証は要りません。

```bash
mkdir -p ~/kukkiri-ltx-v1 && cd ~/kukkiri-ltx-v1
for n in transformer vae prompt; do
  curl -L -O https://huggingface.co/treehog/kukkiri/resolve/main/ltx-v1/kiyo-ltx-v1-$n.safetensors
done
sha256sum kiyo-ltx-v1-*.safetensors
```

Windows（PowerShell）では次のとおりです。
以下、Windows の例は重みを `D:\kukkiri-ltx-v1` に置いたものとして書きます。

```powershell
New-Item -ItemType Directory -Force D:\kukkiri-ltx-v1; cd D:\kukkiri-ltx-v1
foreach ($n in "transformer", "vae", "prompt") {
    curl.exe -L -O "https://huggingface.co/treehog/kukkiri/resolve/main/ltx-v1/kiyo-ltx-v1-$n.safetensors"
}
Get-FileHash kiyo-ltx-v1-*.safetensors -Algorithm SHA256
```

`curl` ではなく `curl.exe` と書きます（Windows PowerShell 5.1 の `curl` は別のコマンドの別名です）。
`Get-FileHash` は SHA-256 を大文字で表示します。

| ファイル | サイズ（バイト） | SHA-256 |
| --- | --- | --- |
| `kiyo-ltx-v1-transformer.safetensors` | 13,364,075,432 | `9712079e1b0ee20a9a47efbd0278cc4fb0eb7b88f9571e9d4d9fdfec70ccc94e` |
| `kiyo-ltx-v1-vae.safetensors` | 1,452,269,922 | `685b06ee3d9b2039647698fc4ea33175112462fc374e2777312c907897dfce8d` |
| `kiyo-ltx-v1-prompt.safetensors` | 8,388,704 | `46f8652e075d12b501f6a21bda80e373dd956aa0f1ce897ca0c82ff7b21af511` |

同じ場所にある `kiyo-ltx-v1-lora.safetensors` は使いません。

## 2. 変換する

jasna の checkout で、jasna の venv から実行します。

```bash
.venv/bin/python scripts/ltx_lora_training/convert_kukkiri_ltx.py ~/kukkiri-ltx-v1 model_weights/ltx-kukkiri
```

Windows:

```powershell
.venv\Scripts\python scripts\ltx_lora_training\convert_kukkiri_ltx.py D:\kukkiri-ltx-v1 model_weights\ltx-kukkiri
```

出力フォルダーには、jasna の 2-step モデル（`--restoration-model-name ltx`）のファイル名で次の 3 つができます。

| ファイル | 内容 |
| --- | --- |
| `ltx-restore-alpha1-distill2-int8.safetensors` | 変換した transformer。kukkiri のプロンプト埋め込みを `prompt_context` として持ち、サンプラーの設定をメタデータに持つ |
| `vae.safetensors` | kukkiri の VAE。ハードリンク、できなければコピー（Windows では、元の重みと出力先が別のドライブだとコピーになり、1.5 GB 増えます） |
| `vae-decoder-full.safetensors` | テンソルのない空のファイル。この VAE は調整済み decoder を使わないが、モデルフォルダーがこの名前を要求する |

スクリプトは最後に、変換が各層の出力に加える相対誤差の最大値を表示します。
0.014 前後なら正常です（Linux で 0.0142、Windows で 0.0144）。
RTX 5080 では 15 秒、RTX 5060 Ti では 40 秒で終わります。

## 3. 実行する

```bash
.venv/bin/python -m jasna --input in.mp4 --output out.mp4 \
    --restoration-model-name ltx --restoration-model-path model_weights/ltx-kukkiri
```

Windows:

```powershell
.venv\Scripts\python -m jasna --input in.mp4 --output out.mp4 `
    --restoration-model-name ltx --restoration-model-path model_weights\ltx-kukkiri
```

LTX の他のオプション（`--ltx-large-canvas`、`--ltx-overlap`、`--ltx-seed`、
`--secondary-restoration rtx-super-res` など）はそのまま使えます。
`--ltx-fast` は使えません（[既知の制限](#既知の制限)）。

### GUI で使う

GUI にはモデルフォルダーを選ぶ欄がなく、LTX のモデルは `model_weights/ltx-restore/` から読みます。
変換の出力先をそこにするか、できた 3 ファイルをそこへ置いてください。

```bash
.venv/bin/python scripts/ltx_lora_training/convert_kukkiri_ltx.py ~/kukkiri-ltx-v1 model_weights/ltx-restore
```

Windows:

```powershell
.venv\Scripts\python scripts\ltx_lora_training\convert_kukkiri_ltx.py D:\kukkiri-ltx-v1 model_weights\ltx-restore
```

Windows で、すでに `model_weights\ltx-kukkiri` へ変換してある場合は、変換し直さずに junction を張れます。
junction は管理者権限なしで作れます。

```powershell
cmd /c mklink /J model_weights\ltx-restore model_weights\ltx-kukkiri
```

外すときは `cmd /c rmdir model_weights\ltx-restore` です。junction だけが消え、変換したファイルは残ります。
`model_weights\ltx-restore` に別のモデルがすでにある場合は、消さずに改名して退避してから張ります。

GUI は、jasna の checkout で起動します（`.venv\Scripts\python -m jasna`、Linux は `.venv/bin/python -m jasna`）。
`model_weights` は起動したフォルダーのものが先に使われるので、別のフォルダーから起動すると、そこに `model_weights` があればそちらを読みます。

復元モデルで「LTX（標準画質）」（2-step）を選びます。
ほかの LTX の画質と「高速モード」（NVFP4）は、対応するファイルが無いので選んでも開始できません。
VAE の encoder と decoder は「自動」のままにします。

開始する前に、「お試しモード（速度テスト）」がオフになっていることを確かめます。
`model_weights/ltx-restore/` にモデルのファイルが無いと、GUI はお試しモードをオンに固定します。
その実行は重みなしの速度テストで、出力は復元されません。
このとき、進捗の表示の末尾に `(trial)` が付きます。
kukkiri の重みで走っているときは `(trial)` が付かず、ログの `settings:` の行に `"weights_source":"kukkiri ..."` が出ます。

## 変換の中身

kukkiri の transformer は、テンソルの構成が jasna の LTX モデルファイルと同じです（48 ブロックと top の層）。
違いは次の 5 点で、変換スクリプトと、ランタイムの 2 つの変更がそれを埋めます。

| 項目 | kukkiri | jasna（upstream のモデル） | 対応 |
| --- | --- | --- | --- |
| キー名 | `model.diffusion_model.transformer_blocks.N.*` | `blocks.N.*` | 変換時に改名 |
| INT8 の形式 | 行ごとの scale を持つ素の INT8 | ConvRot W8A8（入力方向に Hadamard 回転した重みの INT8） | 変換時に逆量子化、回転、再量子化 |
| プロンプト埋め込み | 別ファイルの `video_encoding` | transformer ファイル内の `prompt_context` | 変換時に埋め込む |
| 時間位置のフレームレート | 30 | 24 | メタデータ `position_fps` に書き、ランタイムが読む |
| VAE | LTX-2.5 の conv VAE（`CausalVideoAutoencoder`） | diffusion VAE（`CausalDiffusionVAE`） | ランタイムに conv decoder を追加 |

サンプラーの設定は kukkiri の `ltx.py` のものです（sigmas 1.0 / 0.725 / 0.0 の 2 step、CFG なし、STG なし）。
jasna はこれをモデルファイルのメタデータから読むので、変換時に書き込むだけで済みます。

ランタイムの変更は次の 2 つで、upstream のモデルファイルを読むときの動作は変えません。

- **`position_fps`**（`jasna/ltx/sampler.py`、`jasna/ltx/transformer.py`）: モデルファイルのメタデータに `position_fps` があれば、時間位置をそのフレームレートで計算します。無ければ upstream と同じ 24 です。このキーはこのフォーク独自のものです。
- **設定タグ**（`jasna/settings_tag.py`、`jasna/ltx/model_files.py`）: 変換後のファイルは upstream の 2-step モデルのファイル名で置くので、出力に書く設定タグの `weights` だけでは upstream の重みと区別できません。変換器はメタデータ `converted_from` に出所を書き、ランタイムはそれがあればタグの `weights_release`（upstream のリリース名）の代わりに `weights_source` として書きます。例: `"weights_source":"kukkiri kiyo-ltx-v1-transformer.safetensors (plain int8 -> ConvRot int8)"`。
- **conv VAE**（`jasna/models/ltx_vae/conv_video_decoder.py`、`loader.py`）: VAE の config が `CausalVideoAutoencoder` のとき、encoder は既存のものを使い、decoder は ltx-core から移植した畳み込み decoder を使います。window 全体を 1 回で decode します。

## 状態

- **Linux**（RTX 5080、2026-11-27、v0.12.4 ベース）: 同梱のテストクリップ（1080p、30 fps、10 秒）と、crop が 512 px を超える 1080p の 30 秒素材（899 フレーム、9 window）が完走し、フレーム数は入力と一致しました。30 秒素材の LTX 部分は、既定の 512 px canvas で 40 秒（denoise 0.85 秒 / window-step、decode 0.99 秒 / window）、`--ltx-large-canvas` の 768 px で 102 秒（同 2.37 秒、2.35 秒）でした。VRAM の最大はどちらも 13.3 GB です。LTX は空いている VRAM をブロックの常駐に使うので（48 ブロック中 37）、この値は必要量ではありません。
- **upstream 形式のモデルへの影響なし**: 自前学習のモデル（diffusion VAE）を、この変更の前後のコードで同じ条件で実行した出力は、ビット単位で一致しました。
- **Windows**（RTX 5060 Ti、2026-11-27、v0.12.3 ベース）: 480p から 1080p、24 から 59.94 fps の 8 本（10 秒から 355 秒）が完走し、フレーム数と長さは入力と一致しました。denoise は 1 step 約 2.05 秒でした。利用者の目視では、自前の IC-LoRA モデルよりよく復元できていました。
- **Windows**（RTX 5060 Ti、2026-11-27、v0.12.4 ベース）: 変換は誤差 0.0144、40 秒。同梱のテストクリップが 512 px、768 px（`--ltx-large-canvas`）、`--secondary-restoration rtx-super-res` 併用の 3 通りで完走しました（300 フレーム）。denoise は 512 px で 2.04 秒 / window-step、768 px で 5.65 秒、decode は 2.43 秒 / window と 5.72 秒です。カード全体の VRAM の最大は 512 px で 13.3 GB、768 px で 15.5 GB でした（ほかのアプリの約 2.5 GB を含みます）。GUI では、画面を操作して 1080p の 6242 フレーム（56 window）を RTX Super Res 併用で処理し、LTX 部分は 541 秒でした。出力の設定タグに `weights_source` が入り、利用者の目視で合格しました。
- **diffusion VAE との互換**: kukkiri の transformer の出力は、LTX-2.5 の diffusion VAE でも decode できました。同じクリップで conv decoder の出力との PSNR は 45 dB、所要時間は同じでした。モデルフォルダーの `vae.safetensors` と `vae-decoder-full.safetensors` を diffusion VAE のものに置き換えると、この経路になります。画質の優劣は比べていません。
- テスト: `tests/test_ltx_position_fps_modi.py`, `test_ltx_conv_vae_modi.py`, `test_convert_kukkiri_ltx_modi.py`, `test_settings_tag_modi.py`。

## 既知の制限

- **frozen build では使えません。** 平文のモデルファイルを読むのは開発インストールだけです。
- **`--ltx-fast`（NVFP4）は使えません。** 変換スクリプトは INT8 のファイルだけを作ります。
- **conv decoder は window を分割しません。** diffusion decoder にあるタイル分割と、メモリ不足時の再試行の対象外です。16 GB の GPU では 768 px の canvas でも足りましたが、それより小さい GPU では確認していません。
- **学習時と window の長さが違います。** kukkiri は 97 フレームのクリップで学習していますが、jasna は 121 フレームの window で処理します。静止画では破綻が見えませんでしたが、時間方向への影響は調べていません。
- **再量子化の誤差が乗ります。** 配布されている重みがすでに INT8 なので、変換後の重みは 2 回分の丸めを含みます。

## ライセンスと配布

- `kiyo-ltx-v1` は LTX-2.x の派生物で、[LTX-2.x Community License](https://huggingface.co/treehog/kukkiri/blob/main/ltx-v1/NOTICE.md) に従います。変換後のファイルも同じです。
- 重みも変換後のファイルも、jasna のリポジトリや配布物には含めません。
- 変換スクリプトは、出力のメタデータに元のファイル名とライセンス名を書き込みます。
