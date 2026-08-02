# TensorRT-RTX フレーバー（opt-in、エンジンコンパイルの高速化）

`nvidia-rtx` extra で選べる、TensorRT スタック全体を **TensorRT-RTX**（TensorRT for RTX）に置き換える経路をまとめる。

標準 TensorRT の経路はそのまま既定として残る。
TensorRT-RTX は AOT の全カーネル探索を持たない JIT 方式のため、エンジンビルドが分単位から秒単位になる。
代償は二つある。
実行速度がわずかに劣ること（本書の実測で検出エンジン +12%）と、プロセス起動時に JIT コストを払うことである（検出側はディスクキャッシュで 2 回目以降解消、復元側は毎プロセス数秒が残る）。
初回コンパイルの待ち時間を最優先する利用者向けの選択肢であり、既定を置き換えるものではない。

**実行環境**：本書の数値は Linux / Python 3.13 / torch 2.12.0+cu130 / torch-tensorrt-rtx 2.12.1 / tensorrt-rtx 1.4.0.76 / RTX 5080 (sm120, 16GB) で計測した。
Windows は未検証で、検証手順は引継書（リポジトリ外）に従う。

本実装は姉妹プロジェクト lada-ex の `feat/tensorrt-rtx-migration` ブランチの知見（strongly-typed での `BuilderFlag.FP16` 拒否、フレーバー別エンジンキャッシュの必要性、数値等価性 L1≈0.0005）を土台にしている。

## 導入方法

`nvidia` の代わりに `nvidia-rtx` extra を入れる。

```bash
uv pip install -e .[dev,nvidia-rtx] \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match \
    --prerelease=allow
```

`torch-tensorrt` と `torch-tensorrt-rtx` は同じ `torch_tensorrt` パッケージを提供するため、**両 extra は 1 つの venv に共存できない**。
フレーバーを切り替えたいときは venv を分ける。
torch は同じ 2.12.0+cu130 のままで済み、mmengine パッチ（ビルドガイド §5.1）はこの venv にも必要になる。

インストール後の追加設定は不要である。
フレーバーは `tensorrt_rtx` wheel の有無から自動判別され（`jasna.engine_paths.trt_flavor()`）、CLI と GUI の両方に効く。
緊急時の強制上書きとして環境変数 `JASNA_TRT_FLAVOR`（`rtx` | `standard`）があるが、wheel と食い違う値を指定すると ImportError で早期に落ちる（意図した挙動）。

## エンジンキャッシュの共存

エンジンはフレーバー間で互換性がない（標準 TensorRT のエンジンを TensorRT-RTX ランタイムは読めず、逆も同じ）。
そこで RTX ビルドのエンジンには `.rtx` タグを付けて命名する。

```
rfdetr-v6.bs1-4.fp16.linux.engine        # 標準
rfdetr-v6.bs1-4.fp16.rtx.linux.engine    # RTX
loop_body_backward_1.trt_fp16.rtx.linux.engine
```

標準側の名前は従来と同一なので、既存のエンジンキャッシュはそのまま有効である。
両フレーバーの venv で 1 つの `model_weights` ディレクトリを共有でき、互いのキャッシュを壊さない。
なお RTX エンジンも標準と同じく、同一 OS 内なら GPU をまたいで再利用できる（OS をまたぐ再利用は不可）。

## 内部の差分

TensorRT-RTX は strongly-typed 方式で、精度はネットワークのテンソル dtype で決まる。
このため標準 TensorRT の「fp32 ONNX + `BuilderFlag.FP16` + I/O の HALF 強制」というレシピが使えない。
fp32 の ONNX をそのままパースすると fp32 エンジンになり、実測で 5 倍遅かった（RF-DETR v6、37.6 ms/バッチ4）。

そこで RTX フレーバーでは、パース前に ONNX グラフ全体を fp16 へ変換する（`jasna.trt._convert_onnx_bytes_to_fp16`、依存 `onnxconverter-common`）。
変換は重みと活性化と入出力の全部を対象にし、元グラフに焼き込まれた `Cast(to=float32)` ノード（DINOv2 の埋め込み部に 44 個ある）も fp16 へ書き換える。
この書き換えを省くと、fp16 の重みと fp32 の活性化が混ざって strongly-typed のパースが型不一致で失敗する。

BasicVSR++ 側（`torch_tensorrt.compile(ir="dynamo")` 経路）は import 名が変わらないため無改修で通る。
グラフ分割も標準と同一で、TRT セグメント 4 個と torch 側に残る deform_conv2d 1 個という構造が両フレーバーで一致する。

## JIT コストとディスクキャッシュ

TensorRT-RTX はカーネル生成をエンジンロード時まで遅延する。

生 API 経路（RF-DETR、YOLO）にはディスクキャッシュを実装した（`TrtRunner._create_execution_context`、エンジンの隣に `.jitcache` を置く）。
実測でロード 3.9 秒 → 0.18 秒となり、標準 TensorRT と同水準になる。
キャッシュ生成に失敗しても実行は継続する（プレーンなコンテキスト生成へフォールバック）。

dynamo 経路（BasicVSR++）にはキャッシュがない。
torch-tensorrt-rtx 2.12.1 は `runtime_cache_path` kwarg を受理するが、キャッシュファイルは生成されず、保存済みエクスポートの再ロードも速くならないことを実測で確認した。
このためサブエンジン 6 個のロード JIT（1 プロセスあたり数秒）は毎回発生する。
長尺動画の処理では償却されて消え、短いクリップの繰り返し処理でだけ体感される。

## 実測値（Linux / RTX 5080）

コールドビルドの壁時計時間（コンパイルサブプロセス全体、モデル読み込み込み）。

| エンジン | 標準 TensorRT | TensorRT-RTX | 比 |
|---|---|---|---|
| RF-DETR v6（bs1-4 動的） | 36.1 秒 | 4.8 秒 | 7.6 倍 |
| BasicVSR++ サブエンジン 6 個 | 54.8 秒 | 16.0 秒 | 3.4 倍 |

RTX 5080 は標準 TensorRT のビルドも速い部類なので、絶対値の差は控えめに見える。
GUI の初回実行警告が言う「15〜60 分」級の環境（より遅い GPU、Windows）ほど短縮の絶対量は大きくなるはずで、その確認が Windows 側検証の主目的である。

実行速度と品質。

| 項目 | 標準 TensorRT | TensorRT-RTX |
|---|---|---|
| RF-DETR v6 推論（バッチ4） | 7.5 ms | 8.4 ms（+12%） |
| BasicVSR++ loop_body 1 反復 | 0.31 ms | 0.35 ms |
| e2e 出力の一致（10 秒クリップ） | 基準 | PSNR 46.7 dB / SSIM 0.994 |

e2e の壁時計時間は、10 秒のテストクリップ（300 フレーム）で標準 5.1 秒に対し RTX 11.1 秒だった。
この差の主因は 1 プロセスあたりの復元サブエンジンのロード JIT（前節）で、処理レートそのものの差ではない。
実際、復元が始まる前の区間だけを処理させると両フレーバーとも 0.7 秒で一致し、loop_body 1 反復の差も 0.04 ms にとどまる。
長尺動画ではロード JIT は償却され、定常差は検出エンジンの +12% が上限になる。

CUDA graphs（既定 ON）、`--fp8-recon`（cuDNN FP8 upsample）、torchcodec バックエンドは、いずれも RTX venv で完走を確認した。
graphs の ON/OFF による傾向も標準フレーバーと同じだった。

## 制約と注意点

- torch-tensorrt の RTX 対応は公式に experimental である。
- エンジンは OS をまたいで再利用できない（`.rtx.linux` / `.rtx.win` を別々にビルドする）。
- `tensorrt-rtx` 1.4 系は FP8 エンジンを実行できない既知の問題があるが、jasna は TensorRT の FP8 を使わないため影響しない（`--fp8-recon` は cuDNN 経路で無関係）。
- 検出モデルの fp16 変換は数値をわずかに変える。しきい値近傍の検出が標準フレーバーと異なる可能性は否定できないため、品質が最優先の用途では標準フレーバーを使う。
