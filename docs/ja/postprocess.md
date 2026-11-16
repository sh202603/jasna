# 後処理 CLI `jasna-postprocess`（`+modi`）

`jasna-postprocess` は、完成済みの動画にフレーム全体の後処理だけを掛ける独立コマンドである。
モザイク検出も復元も走らせない。
後処理は次の 2 種類で、片方だけでも両方でも指定できる。

- **全画面アップスケール**（`--upscale`）：jasna 本体の `--upscale` と同じ `span`、`realplksr`、`rtx-super-res`、`dlss5` と、このコマンドだけが持つ `swiftvr`
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
```

`--upscale` と `--frame-generation` のどちらも指定しなければエラーになる。
`--input` がフォルダなら `--output` は出力フォルダで、中の全動画を処理する（`--output-pattern` は本体と同じ意味、既定は `{original}_out`）。
フォルダ内の画像はスキップする。

入力の高さが `--upscale-height` 以上の動画は拡大しない。
フレーム生成も指定していなければ再エンコードせず、フォルダ入力ではその動画をスキップし、単一ファイルではエラーで終了する。

### オプション

| オプション | 既定 | 意味 |
|---|---|---|
| `--upscale` | `none` | `span`、`realplksr`、`rtx-super-res`、`dlss5`、`swiftvr`。 |
| `--upscale-height` | `2160` | `1080`、`1440`、`2160`。幅は縦横比を保つ。 |
| `--upscale-strength` | `1.0` | 0〜1。下げると bicubic の拡大が混ざる。 |
| `--upscale-rtx-*`、`--upscale-dlss5-*` | 本体と同じ | [cli.md](cli.md) を参照。 |
| `--frame-generation` | `none` | `2x`、`4x`。 |
| `--frame-generation-backend` | `rife` | `rife`、`rtx`。本体の既定は `rtx` で、ここだけ違う。 |
| `--frame-generation-rtx-mode` | `medium` | `low`、`medium`、`high`。 |
| `--frame-generation-model-path` | `model_weights/rife.pth` | RIFE の重み。 |
| `--codec` | `hevc` | `hevc`、`av1`。 |
| `--encoder-settings` | （本体の既定） | 本体と同じ形式。 |
| `--device`、`--fp16`、`--batch-size`、`--video-backend` 系、`--log-level`、`--no-progress`、`--disable-ffmpeg-check` | | 本体と同じ意味。 |

SwiftVR 用のオプションは「[SwiftVR の全画面適用](#swiftvr-の全画面適用--upscale-swiftvr)」にまとめる。

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

## 挙動と制約

- `--upscale swiftvr` は `--frame-generation` と併用できない（起動時エラー）。4K の生成器と SwiftVR worker の同居を測っていないためで、必要なら 2 回に分けて実行する。
- 中断からの再開は無い。`swiftvr` の長時間ジョブを止めると最初からやり直しになる。
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
- テスト：`tests/test_postprocess_cli_*.py`、`test_clip_upscale_writer.py`、`test_swiftvr_frame_worker.py`、`test_upscale_swiftvr.py`（いずれも CPU）。
