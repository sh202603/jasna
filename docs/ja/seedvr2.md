# SeedVR2+LoRA 一次復元(`+modi`)

`--restoration-model-name seedvr2` は、一次復元器を BasicVSR++ から
**SeedVR2 3B(one-step SR diffusion)+ LoRA** に差し替える品質モード。
検出された 256px のモザイククロップを、そのまま diffusion forward に通して復元する
(scale=1。二次復元ではなく一次復元の置き換え)。
LoRA は「モザイクをセル平均化の劣化として除去する」ことを学習した自作モデル
(rank 16、約 90 MB、[sh202603/lada-seedvr2-lora](https://huggingface.co/sh202603/lada-seedvr2-lora)、AGPL-3.0)。
SeedVR2 base 重み(Apache-2.0)は
[numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
checkout 側が初回ロード時に自動ダウンロードする。

BasicVSR++ に対して壁時計で約 6 倍遅い(モザイク画面時間に比例。lada-ex 実測で
出力 17〜21 crop-fps、対 BasicVSR++ 100〜157 fps)。生成系のため出力はシャープだが、
「もっともらしいディテール」であって原信号の復元ではない。
低解像度で劣化の大きい素材に最も強く(BasicVSR++ が均す情報の少ない入力からも
diffusion の事前分布がディテールを再構成するため)、BasicVSR++ の完全な置き換え
ではない。一次復元器は素材に応じて使い分ける。
既定モデルは従来どおり basicvsrpp で、本機能は完全なオプトイン。
basicvsrpp 経路の出力は本機能の追加前後で bit 不変。

## 要件

- NVIDIA GPU 16 GB VRAM(全体ピーク実測: 本フォークのスモークで 480p ~12.6 GB。lada-ex 実測は 480p ~12.7 GB / 1080p ~13.9 GB)
- `ComfyUI-SeedVR2_VideoUpscaler` checkout + 専用 venv(ComfyUI 本体は不要)
- ディスク: base 重み ~7.3 GB + LoRA ~90 MB
- BasicVSR++ の TensorRT エンジンは**不要**(初回コンパイルは検出エンジンのみ)

## セットアップ

```bash
# 1. checkout と専用 venv (jasna の venv には何も入れない)
git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git ~/seedvr2_videoupscaler
cd ~/seedvr2_videoupscaler
uv venv --python 3.13 .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt

# 2. LoRA を jasna の model_weights/ へ
wget -O model_weights/lada_seedvr2_lora_v6.pt \
  https://huggingface.co/sh202603/lada-seedvr2-lora/resolve/main/lada_seedvr2_lora_v6.pt

# 3. 実行 (base 重みは初回に checkout 側へ自動ダウンロードされる)
jasna --input in.mp4 --output out.mp4 \
  --restoration-model-name seedvr2 --seedvr2-repo ~/seedvr2_videoupscaler
```

worker の起動(モデルロード + LoRA 注入 + ウォームアップ)は重みキャッシュ済みで
1〜2 分。worker は常駐し、フォルダ入力の複数ファイルを跨いで生き続けるため、
このコストはセッションごとに 1 回だけ払う。

## フラグ

| フラグ | 既定 | 説明 |
|---|---|---|
| `--seedvr2-repo` | — | checkout のパス。seedvr2 選択時は必須 |
| `--seedvr2-python` | `<repo>/.venv/bin/python` | venv の Python(Windows は `Scripts\python.exe`) |
| `--seedvr2-model-dir` | `<repo>/models/SEEDVR2` | base 重みディレクトリ |
| `--seedvr2-dit` | `seedvr2_ema_3b_fp16.safetensors` | DiT 重みファイル名 |
| `--seedvr2-lora` | `<model_weights>/lada_seedvr2_lora_v6.pt` | LoRA checkpoint(fine-tune 品への差し替え口) |
| `--seedvr2-lora-rank` | `16` | LoRA rank(checkpoint と一致必須) |
| `--seedvr2-window` | `33` | スライディングウィンドウ長。4n+1 制約 |
| `--seedvr2-overlap` | `9` | ウィンドウ間クロスフェード幅 |
| `--seedvr2-color-fix` | `lab` | クリップ単位の色補正(`none`/`lab`/`wavelet`) |
| `--seedvr2-empty-cache` | `auto` | 同居する検出/デコード側へのクリップ毎 VRAM 返却。`auto` は総 VRAM 20GiB 未満で `always`(OOM 保険)、以上で `never`(純オーバーヘッドのため) |
| `--seedvr2-accel` / `--no-seedvr2-accel` | off | VAE の融合経路と FP8 DiT(LoRA をマージ)。対応する SeedVR2 checkout が必要。詳細は「[高速化](#高速化--seedvr2-accel)」 |

## 高速化(`--seedvr2-accel`)

`--seedvr2-accel` は worker に二つの部品を要求する。既定は off。

- **VAE の融合経路**(`fused_vae`): GroupNorm と SiLU を 1 回の処理にまとめ、畳み込みを fp16 累積の
  カーネルで実行する(comfy-kitchen)。VAE は fp16 で動く。
- **FP8 DiT**(`fp8_dit`): LoRA を基底の重みへマージしてから、DiT ブロックの線形層を FP8 GEMM に
  差し替える。FP8 化は線形層そのものを置き換えるので、LoRA を側枝として残せない。

どちらも SeedVR2 checkout 側の実装で、上流(numz)の checkout にはまだ無い。worker は起動時に
checkout と GPU を調べ、使えない部品を外して警告を出し、その部品は標準経路で動く
(上流の checkout では両方が外れ、出力は off と bit 一致する)。必要なものは次のとおり。

- この二つの経路を持つ SeedVR2 checkout と、その venv の comfy-kitchen(checkout の
  `requirements.txt` に含まれる)
- `fp8_dit` には RTX 40 系以降の GPU と動く Triton

RTX 5080、256px の合成モザイク、LoRA v7 での worker 単体の実測:

| | off | `--seedvr2-accel` |
|---|---|---|
| 33 フレームのウィンドウ(encode / DiT / decode) | 310 / 252 / 730 ms | 144 / 126 / 342 ms |
| 57 フレームのクリップ | 2953 ms(19.3 crop-fps) | 1590 ms(35.8 crop-fps) |
| worker の VRAM ピーク | 9.6 GiB | 6.0 GiB |

jasna 全体(e2e)では次の値になった。RTX 5080 16 GB の Linux 機で、素材は各 2 分、検出は rfdetr-v6、
LoRA は v6、二次復元はなし。処理時間はプロセスの起動から終了までの壁時計時間で、3 回の中央値である。

| 素材 | 処理時間 off | 処理時間 `--seedvr2-accel` | 倍率 | VRAM ピーク off | VRAM ピーク `--seedvr2-accel` |
|---|---|---|---|---|---|
| 480p | 117.3 秒(30.7 fps) | 69.9 秒(51.5 fps) | 1.68 倍 | 10.9 GiB | 7.6 GiB |
| 1080p | 219.9 秒(16.6 fps) | 125.6 秒(29.1 fps) | 1.75 倍 | 11.7 GiB | 8.1 GiB |
| 4K | 271.7 秒(13.5 fps) | 153.7 秒(23.8 fps) | 1.77 倍 | 13.6 GiB | 9.9 GiB |

VRAM ピークは、GPU 全体の使用量(jasna 本体と worker の合計)から実行前の使用量を引いた値である。
一次復元にかかった時間の比は 3 素材とも 1.81〜1.82 倍で、worker 単体の比(1.86 倍)に近い。
処理時間の倍率がそれより低いのは、起動と終了にかかる 7〜8 秒が変わらないためで、素材が長いほど
1.8 倍に近づく。`--seedvr2-accel` を付けても律速は一次復元のままである(一次復元のスレッドは、
時間の 95% 以上を復元に使っている)。

出力は off と bit 一致しない(worker 単体の条件で PSNR 46.4 dB、最大差 28/255)。上の 3 素材の出力を
目視で比べたところ、off との差は見分けられなかった。LoRA の効果はマージ後も
保たれる(LoRA を効かせない出力との差は 26.5 dB)。lada-ex のハーネスとの bit 一致は、この
オプションを使う間は成り立たない。torch.compile は部品に入れていない(DiT が 126 ms から
101 ms になるだけで、ウィンドウ長ごとに再コンパイルが入るため)。

## 動作の要点

- **ウィンドウ分割**: モデルは短い 4n+1 フレーム列で訓練されているため、クリップは
  33f ウィンドウ、stride 24 で分割して推論し、隣接ウィンドウの重なり 9f を線形ランプで
  GPU 常駐アキュムレータへクロスフェード合成する(CPU への転送はクリップ毎 1 回、量子化は合成後に 1 回)。クリップ境界の継ぎ目は jasna 標準の
  overlap+discard とクロスフェードがそのまま効く。
- **ゼロパディング**: クロップの 256px 化のパディングは、この復元器では reflect でなく
  ゼロ埋めになる(LoRA の訓練分布と一致させるため)。basicvsrpp 経路は従来どおり reflect。
- **色補正**: diffusion 出力はクロップ全体の色がわずかにずれることがあるため、入力
  モザイククロップ自体を参照に LAB 統計(平均と分散)をクリップ一括で転写する。
  `wavelet` は参照の低周波を移植する方式で、モザイクのセル構造を出力へ再導入する
  リスクがあるため既定にしていない。
- **縁の revert**: diffusion の VAE/DiT はゼロパッドの黒を有効領域内 2〜3px へ滲ませ、
  blend がクロップ境界余白を越えて届くため、そのままでは bbox 縁に暗線が出る。
  対策として有効領域のパッド隣接辺 4px(256 空間)を線形ランプで入力へ戻している
  (BasicVSR++ がモザイク外でほぼ恒等なのと同じ状態を作る)。実測で bbox 縁の
  暗線指標は bvpp 同水準まで解消。
- **マスク外 revert**: 1-step diffusion はクロップ全体を再合成するため、暗線がなくても
  blend のフォールオフ環帯には低周波の色・明るさが原画からずれ、粒状感も半減した
  コンテンツが合成され、enlarged bbox が「見えるパッチ」になる。対策として検出マスクの
  信頼ゾーン(マスク + blend の dilation)の外側をすべて入力へ戻す。revert の重みには
  blend mask 自身のフォールオフを流用しており、検出漏れの保険帯は完全に復元されたまま、
  フォールオフ帯は実質原画を合成する。大型モザイクでは信頼ゾーンがクロップ全体を
  覆うため、さらにクロップ境界ランプ(フレーム換算 ~20px = bbox 拡張が保証する
  マスクフリー余白)で全縁のコンテンツを入力へ収束させる。フレーム端にクランプ
  された辺は例外(モザイクがクロップ縁まで達しうる上、そこに継ぎ目は存在しない)。
- **決定性**: seed とアロケータ(cudaMallocAsync)を固定しており、同一入力に対して
  出力は再現的。
- **エラー方針**: 一次復元の失敗はモザイクがそのまま出力に残ることを意味するため、
  素通しフォールバックはない。worker のエラーや死亡は respawn + 同一クリップ 1 回
  リトライ、再失敗はジョブ停止。入出力のフレーム数や形状の不一致は即エラー(リトライなし)。

## 制約と組み合わせ

| 組み合わせ | 挙動 |
|---|---|
| `--vr-mode sbs` / `sbs-fisheye` / `auto` で VR 検出 | 起動時エラー(jasna の VR クロップは射影条件付きで LoRA 未検証) |
| `--frame-gen` | 起動時エラー(常駐 worker と VRAM 共居不能。別パスで実行) |
| `--secondary-restoration flashvsr-inline` | 起動時エラー(worker 2 つで 16 GB 超過) |
| `--secondary-restoration flashvsr`(オフライン) | **可**。Phase 分離により VRAM が両立し、SeedVR2 復元 + FlashVSR 4x の最高品質構成になる |
| `--secondary-restoration tvai` | 警告のみ(生成一次への鮮鋭化二次はエンハンス過剰リスク) |
| `unet-4x` / `rtx-super-res` | 可 |
| `--stream` | 警告のみ(モザイク密集区間は SeedVR2 律速でプレイヤーが停滞し得る) |
| `--fp8-recon` / `--compile-basicvsrpp` / `--restoration-model-path` | BasicVSR++ 専用のため不活性(fp8-recon と model-path は警告) |

## Segment Editor の A/B 比較での利用

Segment Editor の A/B モデル比較でも seedvr2 を選べる。checkout が
`$JASNA_SEEDVR2_REPO`(未設定なら `~/seedvr2_videoupscaler`)にあり、LoRA が
`model_weights/` にあれば、チェックポイント選択に **seedvr2** の項目が現れる。
BasicVSR++ との同一フレーム比較に使える。詳細は [segments.md](segments.md) の
「A/B モデル比較」を参照。

## fine-tune

学習ハーネス(pair dump + LoRA 訓練)は jasna には同梱していない。
dump が依存する BasicVSR++ 訓練用の劣化パイプラインを jasna は vendored していない
(推論専用サブセットのみ)ため、fine-tune は
[lada-ex](https://github.com/sh202603/lada-ex) 側の
`scripts/training/seedvr2_lora/` で行い、生成した checkpoint を `--seedvr2-lora` で
差し替える。手順は lada-ex の `docs/seedvr2_setup.md` を参照。

## 既知の制限

- 生成系の出力は「ありそうなディテール」の付与であり、原信号の復元ではない。
- 画面端で切断された細帯モザイク(実在の検出対象が、正方形化された大きな crop の端の
  30px 程度の帯になる形)は LoRA の訓練分布外で、再構成されず平板なグレーで出ることがある。
  BasicVSR++ は同条件で破綻しないため、同じ区間を既定 restorer で流して比較するのが
  SeedVR2 起因かどうかの最短の切り分けになる。対処は訓練データ側(細帯/端切断帯の分布内化)。
  パイプラインで細帯をスキップする案は実モザイクの放置になるため採らない。
- VR モードは未検証のため起動時に拒否する。対応は今後の評価課題。
- numz repo 内部実装への追従は worker 1 ファイル
  (`jasna/restorer/seedvr2_lora_worker.py`。lada-ex と逐語同一に保つ)に閉じているが、
  上流の互換性破壊時には worker 側の修正が必要になる。
