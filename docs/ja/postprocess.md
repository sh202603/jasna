# 後処理 CLI `jasna-postprocess`（`+modi`）

`jasna-postprocess` は、完成済みの動画にフレーム全体の後処理だけを掛ける独立コマンドである。
モザイク検出も復元も走らせない。
後処理は次の 2 種類で、片方だけでも両方でも指定できる。

- **全画面アップスケール**（`--upscale`）：jasna 本体の `--upscale` と同じ `span`、`realplksr`、`rtx-super-res`、`dlss5` と、`swiftvr`、`relayvsr`（本体の `--upscale` でも使える。「[本体の `--upscale` で使う](#本体の---upscale-で使う)」）
- **フレーム生成**（`--frame-generation 2x|4x`）：RIFE または RTX Video Frame Generation。詳細は [frame_generation.md](frame_generation.md)

`0.12.3+modi` までの `jasna-framegen` を改名して拡張したもので、旧名は残っていない。
引数も本体 CLI の名前に揃えた（「[旧コマンドからの移行](#旧コマンドからの移行)」）。

## 使いどころ

本体の `--upscale` は復元パイプラインの出力段に入っているので、復元と同時にしか使えない。
次の場合はこのコマンドで後から掛ける。

- 復元済みの動画を、復元をやり直さずに拡大したい。
- 本体が `--upscale` との併用を拒否する構成で復元した。`seedvr2`、オフラインと inline の `flashvsr` / `swiftvr`、`--ltx-crop-vsr`、`--segments`、`--ltx-auto-split` が該当する。
- 拡大の設定やコーデックだけを変えて出し直したい。

動画をもう一度エンコードするので、1 パス目は高ビットレートで出しておく（例：`--encoder-settings cq=16`）。

## 使い方

```bash
# パス 1: 復元のみ
jasna --input in.mp4 --output restored.mkv --encoder-settings cq=16

# パス 2: 1080p を 2160p へ
jasna-postprocess --input restored.mkv --output out.mkv --upscale rtx-super-res --upscale-height 2160

# 拡大とフレーム生成を同時に（拡大してから生成する。本体と同じ順序）
jasna-postprocess --input restored.mkv --output out.mkv \
    --upscale rtx-super-res --upscale-height 1440 \
    --frame-generation 2x --frame-generation-backend rtx

# SwiftVR を全画面に掛ける
jasna-postprocess --input restored.mkv --output out.mkv \
    --upscale swiftvr --upscale-height 2160 --swiftvr-repo ~/SwiftVR

# RelayVSR を全画面に掛ける
jasna-postprocess --input restored.mkv --output out.mkv \
    --upscale relayvsr --upscale-height 2160 --relayvsr-repo ~/RelayVSR
```

`--upscale` と `--frame-generation` のどちらも指定しなければエラーになる。
`--input` がフォルダなら `--output` は出力フォルダで、中の全動画を処理する（`--output-pattern` は本体と同じ意味、既定は `{original}_out`）。
フォルダ内の画像はスキップする。

入力の高さが `--upscale-height` 以上の動画は拡大しない。
フレーム生成も指定していなければ再エンコードせず、フォルダ入力ではその動画をスキップし、単一ファイルではエラーで終了する。

### オプション

| オプション | 既定 | 意味 |
|---|---|---|
| `--upscale` | `none` | `span`、`realplksr`、`rtx-super-res`、`dlss5`、`swiftvr`、`relayvsr`。 |
| `--upscale-height` | `2160` | `720`、`1080`、`1440`、`2160`。幅は縦横比を保つ。 |
| `--upscale-strength` | `1.0` | 0〜1。下げると bicubic の拡大が混ざる。 |
| `--upscale-rtx-*`、`--upscale-dlss5-*` | 本体と同じ | [cli.md](cli.md) を参照。 |
| `--frame-generation` | `none` | `2x`、`4x`。 |
| `--frame-generation-backend` | `rife` | `rife`、`rtx`。本体の既定は `rtx` で、ここだけ違う。 |
| `--frame-generation-rtx-mode` | `medium` | `low`、`medium`、`high`。 |
| `--frame-generation-model-path` | `model_weights/rife.pth` | RIFE の重み。 |
| `--codec` | `hevc` | `hevc`、`av1`。 |
| `--encoder-settings` | （本体の既定） | 本体と同じ形式。 |
| `--device`、`--fp16`、`--batch-size`、`--video-backend` 系、`--log-level`、`--no-progress`、`--disable-ffmpeg-check` | | 本体と同じ意味。 |

SwiftVR 用のオプションは「[SwiftVR の全画面適用](#swiftvr-の全画面適用--upscale-swiftvr)」に、RelayVSR 用のオプションは「[RelayVSR の全画面適用](#relayvsr-の全画面適用--upscale-relayvsr)」にまとめる。

### フレーム単位の upscaler

`span`、`realplksr`、`rtx-super-res`、`dlss5` は本体の実装（`jasna/upscale/`）をそのまま使う。
必要なものも本体と同じである。

- `span` と `realplksr` は重みを `model_weights/` に置く。NVIDIA では入力解像度ごとに TensorRT エンジンをコンパイルするので、解像度が混在するフォルダでは複数回コンパイルが走る。
- `rtx-super-res` は NVIDIA 専用で、`nvidia-vfx` が要る。
- `dlss5` は Windows の RTX 30 以降で、初回にランタイムのダウンロードを確認する。

### 旧コマンドからの移行

| `jasna-framegen` | `jasna-postprocess` |
|---|---|
| `--factor 2x` | `--frame-generation 2x`（既定は `none` になった） |
| `--backend` | `--frame-generation-backend` |
| `--rtx-mode` | `--frame-generation-rtx-mode` |
| `--model-path` | `--frame-generation-model-path` |

## SwiftVR の全画面適用（`--upscale swiftvr`）

[SwiftVR](swiftvr.md) は、二次復元で 256px のモザイククロップに使っている one-step の拡散型動画復元モデルである。
`--upscale swiftvr` はこれをフレーム全体に掛ける。
他の upscaler より桁違いに遅く、1080p を 2160p にする場合で RTX 5060 Ti は毎秒約 1.9 フレームである（「[実測](#実測)」）。
短い動画や抜粋に向く。

### 必要なもの

二次復元の SwiftVR と同じ checkout、チェックポイント、venv を使う。
用意の手順は [swiftvr.md](swiftvr.md#必要なもの) にある。

| オプション | 既定 | 意味 |
|---|---|---|
| `--swiftvr-repo` | （必須） | fork `sh202603/SwiftVR` の checkout。 |
| `--swiftvr-python` | `<repo>/.venv` の Python | SwiftVR venv の Python。 |
| `--swiftvr-model-dir` | `<repo>/checkpoints` | チェックポイント。 |
| `--swiftvr-accel` / `--no-swiftvr-accel` | on | FP8 DiT と torch.compile。使えない部品は worker が外して警告する。 |
| `--upscale-swiftvr-tile` | `1.25` | 1 タイルの出力画素数の上限（メガピクセル）。VRAM を決める。 |
| `--upscale-swiftvr-window` | `89` | 1 回の呼び出しで復元するフレーム数。 |

### 仕組み

SwiftVR の VRAM は 1 回の呼び出しの出力画素数で決まり、4K のフレームを 1 枚で処理する構成はコンシューマ GPU に収まらない。
また、jasna が使う `restore_clip()` は呼び出しをまたいで状態を持たない。
そこで、フレームを空間方向にタイルへ、動画を時間方向に窓へ分け、「1 つの窓の 1 つのタイル」を 1 回の呼び出しにする。

- **タイル**：フレームを均等な長方形に割る。枚数は、1 タイルの出力が `--upscale-swiftvr-tile` 以下になる最小の数で、隣り合うタイルは出力で 96px 以上重なる。1080p の 2 倍では 4 列 2 行の 8 タイル（各 528×576）になり、処理する画素はフレームの 1.12 倍である。重なった部分は線形の重みで混ぜる。
- **窓**：89 フレームを 1 窓とし、隣の窓と 16 フレーム重ねる。重なりの先頭 5 フレームは前の窓、末尾 5 フレームは次の窓の結果を使い、中央の 6 フレームを線形に混ぜる。窓の先頭は過去の文脈なしで生成されるので、その部分を出力に入れないための重なりである。本体の clip の境界と同じ数値にしてある。
- **倍率**：SwiftVR は 2 倍か 4 倍しか出せない。目標の高さが入力の 2.5 倍以下なら 2 倍、それを超えれば 4 倍で生成し、端数は bicubic のリサイズで合わせる。1080p から 1440p は 2 倍で生成して縮めるので、出力の画素数に対して生成が 2.25 倍多い。
- **色補正**：混ぜ終えたフレーム全体に、入力フレームの bicubic 拡大を参照とする wavelet 補正を掛ける。二次復元の SwiftVR と同じ関数で、常に有効である（検証用の上書きは `JASNA_SWIFTVR_COLOR_FIX`）。

タイルと色補正は SwiftVR venv で動く worker（`jasna/restorer/swiftvr_frame_worker.py`）が行い、窓の重ねとクロスフェードは jasna 側の writer（`jasna/upscale/clip_writer.py`）が行う。
worker は 1 回だけ起動し、フォルダ入力では全動画で使い回す。
フレームサイズや倍率が変わると、torch.compile の再コンパイルがその動画の最初の窓で走る。

### 実測

Windows 11、RTX 5060 Ti 16 GB、2026-11-17。
SwiftVR fork `137de20`、FP8 と torch.compile あり、タイルと窓は既定値。
GPU 全体のピークは `nvidia-smi` の 0.5 秒ポーリングの最大値で、デスクトップ常駐（約 3.5 GB）を含む。

| 入力 | 出力 | 倍率とタイル | 1 窓の時間 | GPU 全体のピーク |
|---|---|---|---|---|
| 1080p、180 フレーム | 3840×2160 | 2 倍、4×2（528×576） | 89 フレームで復元 33.7 秒 + 合成と転送 5.5 秒 | 15.3 GB |
| 720p、60 フレーム | 2560×1440 | 2 倍、2×2（672×384） | 60 フレームで復元 10.3 秒 + 合成と転送 1.9 秒 | 未計測 |
| 480p、60 フレーム | 2562×1440 | 4 倍、3×2（304×256） | 60 フレームで復元 49.5 秒（再コンパイルを含む）+ 合成と転送 2.9 秒 | 未計測 |

- 1080p から 2160p は、1 窓が約 39 秒で 73 フレーム進むので、定常で毎秒約 1.9 フレームになる。30 fps の 1 分に約 16 分かかる。
- worker の起動は、モデルの読込が約 6.5 秒、warmup が 7〜41 秒だった。warmup の幅は torch.compile のキャッシュの有無による。
- GPU 全体のピーク 15.3 GB は 16 GB の上限に近い。デスクトップの常駐が大きい環境や 12 GB 級では `--upscale-swiftvr-tile` を下げる（タイルが増えて遅くなる）。
- 出力のフレーム数と音声のパケット数は、3 本とも入力と一致した。
- 720p と 480p の行は 1 回のフォルダ入力で続けて処理したもので、倍率とタイル形状が変わっても同じ worker で完走した。

1080p の出力で継ぎ目を数値で調べた。

- **窓の境界**：出力のフレーム間差を入力のフレーム間差で割った比は、重なりの区間で平均 0.995（最大 1.009）、それ以外で平均 0.997（最大 1.021）だった。境界での跳ねは無い。この指標は被写体の動きが支配的な粗いもので、質感の食い違いまでは測れない。
- **タイルの境界**：入力の bicubic 拡大との差（生成された細部の量）は、重なりの帯 4 本のうち 3 本で周囲より約 9% 小さく、残り 1 本は周囲と同じだった。2 つのタイルの生成結果を平均するので、帯の中は細部がやや減る。

利用者がこの 1080p から 2160p の出力を目視し、問題なしと判断した。
別の実素材 1 本（720p から 1080p、1763 フレーム）も毎秒 4.3 フレームで完走し、仕上がりは良好だった。

## RelayVSR の全画面適用（`--upscale relayvsr`）

[RelayVSR](https://github.com/kopperx/RelayVSR) は、ストリーミング型の動画超解像モデルである。
大きいモデル（Wan）が数フレームおきのキーフレームを復元し、小さいデコーダーがキーフレームの間のフレームを埋める。
`--upscale relayvsr` はこれをフレーム全体に掛ける。
RTX 5060 Ti で 1080p を 2160p にする場合、既定の設定で毎秒約 1.7 フレームである（「[RelayVSR の実測](#relayvsr-の実測)」）。

### RelayVSR に必要なもの

RelayVSR は jasna と別の torch（2.10、CUDA 12.8）に固定されているので、自分の venv で動かす。

1. fork [sh202603/RelayVSR](https://github.com/sh202603/RelayVSR)（`windows-support` ブランチ）を clone する。上流には jasna が使う `build_stream()` と 2 倍の重みの対応が無い。
2. checkout の中で `uv sync --locked --extra cu128` を実行する。
3. 重みを `weights/` に置く。倍率ごとに別の組で、使う倍率の組だけあればよい。

| 倍率 | ファイル | 入手 |
|---|---|---|
| 2 倍 | `wan_x2.safetensors`、`flash_x2.safetensors` | `uvx hf download lightx2v/RelayVSR_X2 wan_x2.safetensors flash_x2.safetensors --local-dir weights` |
| 4 倍 | `wan.safetensors`、`flash.safetensors` | `uvx hf download kopper/RelayVSR wan.safetensors flash.safetensors --local-dir weights` |

| オプション | 既定 | 意味 |
|---|---|---|
| `--relayvsr-repo` | （必須） | fork `sh202603/RelayVSR` の checkout。 |
| `--relayvsr-python` | `<repo>/.venv` の Python | RelayVSR venv の Python。 |
| `--relayvsr-weights-dir` | `<repo>/weights` | 重みの置き場所。 |
| `--relayvsr-window-size` | `16` | キーフレームの間隔 + 1。大きいほど速いが、動きの速い場面で入力から離れやすい。 |
| `--relayvsr-wan-weights` | `fp8` | `fp8`、`bf16`。`fp8` は RTX 40 以降が要り、使えなければ警告して `bf16` で動く。`bf16` は RelayVSR の基準の出力になる。 |
| `--relayvsr-kv-cache` | `auto` | `auto`、`gpu`、`cpu`。Wan の K/V キャッシュの置き場所。 |
| `--relayvsr-kv-cache-dtype` | `native` | `native`、`int8`。`int8` はキャッシュを半分にし、出力がわずかに変わる。 |

### RelayVSR の仕組み

- **状態は動画全体で続く**：RelayVSR は Wan の K/V キャッシュを動画の最後まで持ち続ける。SwiftVR のように動画を窓へ切り直すと文脈を捨てることになるので、タイルも窓の重ねもクロスフェードも使わず、フレームを 1 枚ずつ順に渡す。
- **出力は遅れて出る**：フレームは、その後ろのキーフレームができてから出力される。最初は 16 フレームを読んでから 16 フレームがまとめて出て、以後は 15 フレームごとに 15 フレームが出る（`--relayvsr-window-size 16` のとき）。jasna 側の writer（`jasna/upscale/stream_writer.py`）は、まだ出てきていないフレームの pts を順に持っておく。
- **倍率**：RelayVSR は 2 倍か 4 倍しか出せず、倍率は重みの組で決まる。選び方は SwiftVR と同じで、目標の高さが入力の 2.5 倍以下なら 2 倍、それを超えれば 4 倍で生成し、端数は bicubic のリサイズで合わせる。
- **メモリは生成する画素数で決まる**：支配的なのは K/V キャッシュで、2160p を生成するとき bf16 で約 12 GB になる。`auto` は動画の開始時に GPU の空きを見て、収まらなければホストメモリに置く（出力は同じで、キーフレームが遅くなる）。
- **2160p を超える生成**：720p から 2160p は 4 倍で 5120×2880 を生成して縮める。1440p から 2160p も 2 倍で同じ大きさになる。キャッシュは 2160p の約 1.8 倍になるので、該当する動画では開始時に警告を表示する。この組み合わせは実測していない。

worker（`jasna/restorer/relayvsr_frame_worker.py`）は RelayVSR venv で 1 回だけ起動し、フォルダ入力では全動画で使い回す。
倍率が変わる動画では、worker がもう一方の重みの組を読み直す（本機で約 8 秒）。

### RelayVSR の実測

Windows 11、RTX 5060 Ti 16 GB、2026-11-17。
RelayVSR fork `77be23e`。
速度は worker が 1 本の動画に掛けた時間から出したもので、デコード、転送、エンコードを含む。
GPU 全体のピークは `nvidia-smi` の 0.5 秒ポーリングの最大値で、デスクトップ常駐（約 3.0 GB）を含む。

1920×1080 の 151 フレームを 3840×2160 にした結果（2 倍）。

| 設定（既定からの変更） | 速度 | K/V キャッシュ | GPU 全体のピーク |
|---|---|---|---|
| なし（`fp8`、`auto`、`native`、窓 16） | 1.69 fps | ホストメモリ | 13.2 GiB |
| `--relayvsr-window-size 32` | 2.38 fps | ホストメモリ | 11.9 GiB |
| `--relayvsr-kv-cache-dtype int8` | 0.72 fps | GPU（`auto` が選択） | 15.6 GiB |
| `--relayvsr-kv-cache gpu --relayvsr-kv-cache-dtype int8` | 0.73 fps | GPU | 15.6 GiB |
| 上に加えて `--relayvsr-window-size 32`、`--upscale-strength 0.8` | 0.86 fps | GPU | 15.5 GiB |

- **本機では既定が最速である**：K/V キャッシュを GPU に置いた 3 行は、GPU 全体が 16 GB の上限に達して既定より遅くなった。`int8` を付けただけの行も、キャッシュが小さくなったことで `auto` が GPU を選び、同じ結果になった。`auto` は動画の開始時の空きしか見ないので、その後に増える分（jasna 側のエンコーダーなど）とデスクトップの常駐が大きい環境では見積もりが外れる。RelayVSR の README が 16 GB 向けに勧める `fp8`、`gpu`、`int8` の組は、本機の jasna 経由では勧められない。
- **窓を大きくすると速いが、入力から離れる**：出力を 1080p に縮めて入力と比べた PSNR は、窓 16 で平均 34.3 dB（最小 26.8 dB）、窓 32 で平均 29.7 dB（最小 20.4 dB）だった。下がるのは動きの速い区間である。素材 1 本だけの数値で、見た目の良し悪しは判定していない。
- **フォルダ入力**：1440×1080（2 倍、1.56 fps）と 720×540（4 倍、2.12 fps）の各 48 フレームを続けて処理し、倍率が変わっても同じ worker で完走した。どちらも 2880×2160 を生成し、ピークは 13.9 GiB だった。
- **起動**：重みの読込は約 7 秒、worker の起動から最初の動画の開始までは約 10 秒だった。
- **worker を通しても画素は変わらない**：360×270 の 33 フレームで、RelayVSR の `build_stream()` を直接回した結果と worker 経由の結果が全フレーム、全画素で一致した。
- 出力のフレーム数と音声のパケット数は、すべての実行で入力と一致した。フレームを 1 枚ずらして比べると PSNR は 23.1 dB に落ちるので、フレームの順序も入力と合っている。

### SwiftVR との比較

同じ日、同じ機材で、ネイティブ 1080p の復元済み動画 180 フレームを両方で 2160p にした。
エンコードはどちらも `cq=16` で、RelayVSR の GPU 全体のピークが上の表より低いのは、デスクトップの常駐が約 1.75 GiB だったためである。

| | SwiftVR（既定） | RelayVSR（既定） | RelayVSR 窓 32 | RelayVSR `bf16` |
|---|---|---|---|---|
| 処理速度（起動を除く） | 1.81 fps | 1.74 fps | 2.38 fps | 0.79 fps |
| 起動（読込と warmup） | 34.9 秒 | 7.1 秒 | 7.1 秒 | 6.4 秒 |
| 全体の時間 | 155 秒 | 119 秒 | 91 秒 | 244 秒 |
| GPU 全体のピーク | 13.3 GiB | 12.1 GiB | 12.3 GiB | 15.2 GiB |
| worker のホストメモリのピーク | 19.5 GiB | 17.7 GiB | 17.7 GiB | 19.6 GiB |
| 入力との一致（出力を縮めた PSNR、平均 / 最小） | 37.3 / 32.9 dB | 36.4 / 33.8 dB | 33.8 / 18.5 dB | 36.3 / 33.8 dB |
| 鮮鋭度（ラプラシアン分散、bicubic 比） | 41.9 | 8.1 | 7.5 | 8.3 |

- **速度と VRAM は既定同士でほぼ同じである**：差は起動で、RelayVSR は torch.compile の warmup が無い分だけ約 28 秒早く始まる。
- **ホストメモリはどちらも 20 GiB 近く使う**：RelayVSR は K/V キャッシュがホスト側に置かれるため、SwiftVR は 1 窓分のタイルの結果を CPU に持つためである。
- **高周波の量は SwiftVR が約 5 倍多い**：数値で最も大きい違いである。入力への一致は同程度だった。
- **窓 32 は一部のフレームが入力から大きく離れる**：最小が 18.5 dB まで落ちた。既定の 16 を勧める。
- **`bf16` は 16 GB では遅い**：VRAM が上限近くまで埋まり、速度が半分以下になった。指標は `fp8` とほぼ同じである。

利用者がこの比較の出力を目視し、SwiftVR と RelayVSR の仕上がりは同等と判断した。

## 本体の `--upscale` で使う

`jasna --upscale swiftvr` と `jasna --upscale relayvsr` は、同じ worker と writer を復元パイプラインの出力段に入れる。
復元と拡大が 1 回の実行で済み、中間の動画とその再エンコードが要らない。

```bash
jasna --input in.mp4 --output out.mp4 \
    --upscale relayvsr --upscale-height 1080 --relayvsr-repo ~/RelayVSR
jasna --input in.mp4 --output out.mp4 \
    --upscale swiftvr --upscale-height 1080 --swiftvr-repo ~/SwiftVR
```

- **オプション**：`--swiftvr-repo`、`--swiftvr-python`、`--swiftvr-model-dir`、`--swiftvr-accel` と `--relayvsr-*` は二次復元と同じ引数を使う。`--upscale-swiftvr-tile` と `--upscale-swiftvr-window` はこのコマンドと同じ意味である。
- **併用できないもの**：`--secondary-restoration`（`none` 以外すべて）と `--frame-generation`。worker が復元と同じ GPU に同居するので、VRAM を分け合う相手を置かない。`--upscale` 全般の制約（`--segments`、`--ltx-auto-split`、`--stream`、`seedvr2`、`--ltx-crop-vsr`）もそのまま掛かる。
- **worker の寿命**：最初に拡大する動画で起動し、フォルダ入力では後続の動画で使い回す。
- **GUI**：選べない。CLI だけである。

### 同居の実測

Linux、RTX 5080 16 GB、2026-11-28。
一次復元は BasicVSR++（二次復元なし）、入力は 640×480 の 1,950 フレーム（65 秒）、worker の設定は既定値である。
壁時計はプロセスの起動から終了までで、エンジンとモデルの読込を含む。
GPU 全体のピークは `nvidia-smi` の 0.5 秒ポーリングの最大値で、デスクトップの常駐（約 2.3 GiB）を含む。

| `--upscale` | 出力 | 壁時計 | GPU 全体のピーク |
|---|---|---|---|
| なし | 640×480 | 11.5 秒 | 5.2 GiB |
| `rtx-super-res` | 1440×1080 | 12.4 秒 | 5.6 GiB |
| `swiftvr` | 960×720 | 92.7 秒 | 14.4 GiB |
| `swiftvr` | 1440×1080 | 71.6 秒 | 14.5 GiB |
| `relayvsr` | 1440×1080 | 64.7 秒 | 13.0 GiB |
| `swiftvr` | 2880×2160 | 303.7 秒 | 15.0 GiB |
| `relayvsr` | 2880×2160 | 219.7 秒 | 14.0 GiB |

- 7 ケースとも完走し、出力は全件 1,950 フレームだった。
- `swiftvr` の 2160p は 16 GB の上限に達し、jasna 側の VRAM 退避が 21 回（計 697 MiB）働いた。それ以外のケースでは退避は起きていない。余裕が無い環境では `--upscale-swiftvr-tile` を下げるか、このコマンドで 2 パスに分ける。
- `relayvsr` の K/V キャッシュは、1080p では GPU、2160p ではホストメモリに置かれた（`auto`）。
- `swiftvr` の 720p は 1080p と同じ 2 倍（1280×960）を生成して縮めるので、速くならない。720p の壁時計が長いのは、最初のケースで worker の warmup が長かった（26 秒）ためである。
- 画質と 1080p 入力は測っていない。

### Windows の実測

Windows、RTX 5060 Ti 16 GB、2026-11-28。
入力は 720×480 の 890 フレームと 1920×1080 の 899 フレーム（どちらも約 30 秒）、worker の設定は既定値である。
LTX は kukkiri の重みで、どちらの素材も window は 9 個である。
GPU 全体のピークは同じ 0.5 秒ポーリングの最大値で、ほかのアプリの 1.2〜2.2 GiB を含む。

| 一次 | 入力 | `--upscale` | 出力 | 壁時計 | うち compose | GPU 全体のピーク |
|---|---|---|---|---|---|---|
| BasicVSR++ | 720×480 | なし | 720×480 | 39 秒 | | 4.7 GiB |
| BasicVSR++ | 720×480 | `rtx-super-res` | 1080×720 | 33 秒 | | 5.0 GiB |
| BasicVSR++ | 720×480 | `relayvsr` | 1620×1080 | 117 秒 | | 13.4 GiB |
| BasicVSR++ | 720×480 | `swiftvr` | 1620×1080 | 167 秒 | | 13.0 GiB |
| LTX | 720×480 | なし | 720×480 | 112 秒 | 25.2 秒 | 14.6 GiB |
| LTX | 720×480 | `relayvsr` | 1080×720 | 190 秒 | 117.1 秒 | 13.3 GiB |
| LTX | 720×480 | `relayvsr` | 1620×1080 | 173 秒 | 99.4 秒 | 13.2 GiB |
| LTX | 720×480 | `swiftvr` | 1620×1080 | 194 秒 | 121.5 秒 | 14.1 GiB |
| LTX | 720×480 | `swiftvr` | 3240×2160 | 394 秒 | 321.8 秒 | 15.1 GiB |
| LTX | 720×480 | `relayvsr` | 3240×2160 | 398 秒 | 324.5 秒 | 15.1 GiB |
| LTX | 1920×1080 | `relayvsr` | 3840×2160 | 534 秒 | 456.9 秒 | 15.3 GiB |
| LTX、768 px の window | 1920×1080 | `relayvsr` | 3840×2160 | 635 秒 | 483.3 秒 | 15.5 GiB |

- 12 ケースとも完走し、出力のフレーム数は入力と一致した。
- compose の秒数は、同じ素材の Linux（RTX 5080）の 2.1〜2.6 倍だった。「なし」に対する比は `relayvsr` の 1080p で 3.9 倍で、Linux の 4.0 倍と同じである。
- LTX 一次の `relayvsr` の 2160p は、Linux では 720×480（4 倍）と 1920×1080（2 倍）のどちらも OOM で失敗したが、Windows では完走した。違いの原因は調べていない。
- LTX 一次で 1920×1080 を `swiftvr` で 2160p にするケースは、10 分で打ち切った（Linux では 277.7 秒で完走）。エラーは出ていない。
- `relayvsr` の K/V キャッシュは、BasicVSR++ 一次では GPU、LTX 一次ではホストメモリに置かれた。
- フォルダ入力（BasicVSR++ 一次、`relayvsr` の 1080p、2 本）では、worker は 1 回だけ起動し、2 本目で使い回された。
- 利用者が 720×480 入力の出力（BasicVSR++ 一次と LTX 一次、`relayvsr` と `swiftvr`）を目視し、問題は無く Linux の出力と同等と判断した。画質の数値評価は行っていない。

### LTX 一次との同居

LTX では、worker は compose パスで LTX window の VAE デコードと同じ GPU を使う。
両方のピークが重なると 16 GB では足りないので、次の 2 つを行う。

- **交互に使う**：window のデコードと worker の処理は同時に走らない。デコードは終わるたびに確保した VRAM を返す（`jasna/ltx/compose_pass.py` の `share_gpu_turn`）。
- **RelayVSR の K/V キャッシュ**：`--relayvsr-kv-cache auto` は、VRAM が 20 GB 未満のカードではホストメモリに置く。worker は動画の先頭で置き場所を決めるが、その時点では後のデコードが使う VRAM が見えないためである。`gpu` を明示すれば GPU に置く。

同じ機材と素材、一次は kukkiri の重みの LTX（512 px の window が 22 個）、2026-11-28。

| `--upscale` | 出力 | 壁時計 | うち compose | GPU 全体のピーク |
|---|---|---|---|---|
| なし | 640×480 | 84.3 秒 | 23.2 秒 | 13.4 GiB |
| `relayvsr` | 1440×1080 | 137.1 秒 | 80.6 秒 | 14.0 GiB |
| `swiftvr` | 1440×1080 | 158.3 秒 | 93.8 秒 | 14.1 GiB |
| `swiftvr` | 2880×2160 | 367.3 秒 | 308.6 秒 | 15.3 GiB |
| `relayvsr` | 2880×2160 | 失敗 | | 15.4 GiB |

- 「なし」のピークは denoise のもので、LTX は空いている VRAM を transformer のブロックの常駐に使う。
- `relayvsr` の 2160p（4 倍、2560×1920 を生成）は、481 フレーム目で jasna 側が OOM になった。その時点で jasna が 6.1 GiB、worker が 6.9 GiB を使っていた。この組み合わせは開始時に警告を出す。高さを下げて 2 倍にするか、`swiftvr` を使うか、このコマンドで 2 パスに分ける。
- 上の 2 つの対策を入れる前は、`relayvsr` の 1080p だけが完走し、他の 3 ケースは OOM で失敗した。
- この機材では、768 px の window（`--ltx-large-canvas`）と画質は測っていない。1080p 入力と 768 px の window の Windows での実測は「[Windows の実測](#windows-の実測)」にある。

## 挙動と制約

- `--upscale swiftvr` と `--upscale relayvsr` は `--frame-generation` と併用できない（起動時エラー）。4K の生成器と worker の同居を測っていないためで、必要なら 2 回に分けて実行する。
- 中断からの再開は無い。`swiftvr` や `relayvsr` の長時間ジョブを止めると最初からやり直しになる（RelayVSR の `--resume` は RelayVSR のコマンドの機能で、ここからは使えない）。
- `relayvsr` は Linux で未検証である。
- 出力に設定コメントのタグは書かない。
- LUT、シャープ、fps のリターゲットなど、本体の出力オプションは持たない。
- CLI のみで、GUI には出ない。
- `jasna.pipeline` と `jasna.protection` を import しない（テストで保証）。
- `span`、`realplksr`、`dlss5` は重みとランタイムが手元に無く、このコマンドでも実走行は未検証である（配線はテストで確認）。frozen build もこの変更後は未検証である。

## 実装

- `jasna/postprocess_cli.py`：引数、検査、writer の組み立て。`UpscaleSettings` は `session_factory` の全画面アップスケール関数が `SessionConfig` から読む属性だけを持ち、その代わりに渡す。
- `jasna/upscale/clip_writer.py`：`ClipUpscaler` プロトコルと `ClipUpscaleWriter`（窓、重なり、クロスフェード）。
- `jasna/upscale/swiftvr.py`：`SwiftvrWorker`（worker の起動と wire）、`SwiftvrClipUpscaler`、倍率の選択。
- `jasna/restorer/swiftvr_frame_worker.py`：SwiftVR venv で動く worker（タイルの割り付け、合成、色補正）。モデルの扱いは `swiftvr_inline_worker.py` を path で読んで共有する。
- `jasna/upscale/stream_writer.py`：`StreamUpscaler` プロトコルと `StreamUpscaleWriter`（遅れて出るフレームと pts の対応）。
- `jasna/upscale/relayvsr.py`：`RelayvsrWorker`（worker の起動と wire）、`RelayvsrStreamUpscaler`。
- `jasna/restorer/relayvsr_frame_worker.py`：RelayVSR venv で動く worker。動画のフレームを stdin から読むジェネレーターとして `stream.frames()` に渡す。wire の定義はこのファイルの冒頭にある。
- `jasna/restorer/relayvsr_common.py`：`--relayvsr-*` のパスの検査と、倍率ごとの重みの組。
- テスト：`tests/test_postprocess_cli_*.py`、`test_clip_upscale_writer.py`、`test_swiftvr_frame_worker.py`、`test_upscale_swiftvr.py`、`test_stream_upscale_writer.py`、`test_upscale_relayvsr.py`、`test_relayvsr_frame_worker.py`（いずれも CPU）。
