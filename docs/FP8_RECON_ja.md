# cuDNN FP8 復元バックエンド（実験的、TensorRT フォールバック付き）

`--fp8-recon` で選べる、BasicVSR++ の upsample サブエンジンを cuDNN graph API の FP8 畳み込みで置き換える経路の設計をまとめる。

既存の TensorRT FP16 サブエンジン経路はそのまま残す。
FP8 バックエンドの構築に失敗した場合（非対応 GPU、依存欠落など）は、警告を出して自動的に TensorRT エンジンへフォールバックする。
既定は無効で、これは従来と完全に同一の挙動になる。

**実行環境**：jasna 本体と同じ GPU 専用スタックに加え、依存 `nvidia-cudnn-frontend`（1.25.0 で検証）、cuDNN 9.17 以上（torch 2.12.0+cu130 同梱の 9.20 で充足）、FP8 対応 GPU（sm89 以上、RTX 40 系以降）を要する。
本書の数値は Linux / Python 3.13.14 / torch 2.12.0+cu130 / TensorRT 10.16.1.11 / cuDNN 9.20 / RTX 5080 (sm120, 16GB) / `--max-clip-size 90` で計測した。
Windows でも検証済みで、Windows 11 / 同一 RTX 5080 上で同じ A/B ベンチマークが同等の数値で全ゲートを通過する（速度比 1.42〜1.56 倍、PSNR 64.1 dB、SSIM 0.99976、VRAM 純減 1976 MB、ビット決定的。glue は triton-windows wheel 経由でコンパイル）。詳細は「制限と注意点」を参照。

本実装は姉妹プロジェクト lada-ex の `feat/fp8-recon` ブランチ（`lada/models/basicvsrpp/fp8_recon.py`、AGPL-3.0）からの移植である。
lada-ex で reconstruction と呼ばれるステージは、jasna の upsample サブエンジンと同一の部分ネットワークを指す。

## 背景と目的

復元パイプラインの主役である BasicVSR++ は、TensorRT FP16 のサブエンジン群として実行される。
このうち upsample サブエンジン（入力特徴の再構成畳み込み 11 層と、pixel shuffle による 4 倍拡大の尾部 5 層）は、動的プロファイル（クリップ長 1〜`max_clip_size`）を持つため、**ロード時に約 2.2GB の内部アリーナを確保する**。
この確保はロード時に一度だけ起こり、実行中ずっと保持される。

一方、cuDNN 9.17 以降は Blackwell ネイティブ（sm120）の FP8 畳み込みカーネルを持ち、この形状群で最良 FP16 カーネル比 2.1〜3.0 倍を実測した。
TensorRT 側の FP8 は袋小路である。
標準 TensorRT（10.x / 11.1）と TensorRT-RTX 1.5 のいずれでも FP8 畳み込みは 1.01〜1.08 倍にとどまることを lada-ex 側で実測済みで、TensorRT の precision 指定で再挑戦する価値はない。

そこで upsample サブエンジンだけを cuDNN graph API の FP8 実装で置き換える。
狙いは二つある。
ステージ単体の高速化（実測 1.5 倍前後）と、TensorRT エンジンのロード自体をスキップすることによる約 2GB の VRAM 解放である。
後者が実運用上の主目的になる（評価結果の節で述べるとおり、パイプラインの律速は検出側にあり、fps には現れないため）。

## アーキテクチャ

### 対象と接合点

置き換えるのは `BasicVSRPlusPlusNetSplit` が呼ぶ upsample エンジン 1 呼び出しで、入出力は `(T, 5·mid, 64, 64)` fp16 から `(T, 3, 256, 256)` fp16 である（mid は `generator.mid_channels`、既定 64）。
残差加算（`+ lqs`）はエンジンの外（呼び出し側）にあり、FP8 バックエンドも加算前の値を返す。
lada-ex 版は加算をエンジン内に持つため、この点が移植時の主要なインターフェース差分になる。

構築と注入は `create_split_forward()`（`jasna/restorer/basicvsrpp_sub_engines.py`）で行う。

1. 環境変数 `JASNA_FP8_RECON=1` かつ fp16 かつ CUDA デバイスのときだけ、`CudnnFP8Upsample`（`jasna/restorer/fp8_upsample.py`）の構築を試みる。
2. 構築は TensorRT サブエンジン群のロードより先に行う。FP8 の warmup と torch.compile のコンパイルを、TensorRT のアリーナ確保前に済ませるためである。
3. 構築に成功したら、`load_sub_engines(..., load_upsample=False)` で TensorRT upsample エンジンのロード自体をスキップする。ロードしてから解放するのではなくロードしないので、起動時からアリーナが確保されない。
4. 構築に失敗したら（例外はすべて捕捉）、警告ログを出して従来どおり TensorRT エンジンをロードする。

エンジンファイルの存在チェック（コンパイル、preflight、`load_sub_engines` の入口）は変更しない。
TensorRT upsample エンジンはフォールバック先として引き続きビルド・保持される。

解放は `BasicVSRPlusPlusNetSplit.close()` が担い、エンジンが `release()` を持つ場合はそれを、持たない場合は従来の TensorRT 解放処理を呼ぶ。
`CudnnFP8Upsample` は（パラメータを持たない）`nn.Module` として実装し、`close()` の `modules()` 走査と属性クリアに耐える形にしてある。

### 数値設計

数値設計は lada-ex の検証結果をそのまま引き継ぐ。

- **全鎖 FP8/NHWC**：入口で fp16 から FP8 (e4m3) へ 1 回だけキャストし、畳み込み間は FP8 のまま通す。中間を fp16 に戻すと利得が消える。
- **融合 epilogue 3 種**：`act`（bias、ReLU または LeakyReLU(0.1)、スケール乗算、FP8 出力）、`res`（スケール乗算、bias、恒等加算、FP8 出力）、`final`（スケール乗算、bias、fp16 出力）。畳み込み後の処理をすべて cuDNN graph に融合する。
- **スケーリング**：weight は per-tensor の amax スケール（epilogue に折り込み）、活性化は scale 1.0 固定。活性化の amax は数単位で FP8 の上限 448 より十分小さく、キャリブレーションは不要である。この成立は ReLU / LeakyReLU が正のスケーリングと可換であること（正斉次性）に依存しており、活性化関数を変えるなら再検討を要する。
- 計算は `intermediate_data_type=FLOAT` / `compute_data_type=FLOAT`、ヒューリスティクスは `[A, FALLBACK]`。

### 実装の要点

- **手動 NHWC pixel shuffle**：torch ネイティブの `pixel_shuffle` は channels-last の 1 バイトテンソルで遅い gather 経路に落ち、約 4 倍遅い。int8 ビューでの軸入れ替えによる手動実装を使う。
- **T バケットと尾部タイル**：クリップ長 T は 10 の倍数に切り上げてバケット化し、バケットごとに cuDNN graph を持つ。尾部（pixel shuffle 以降の 128²/256² を扱う 5 層）はフレーム独立なので T=10 のタイルで回し、大きなバッファをタイルサイズに抑える。これが FP8 バックエンドの常駐 VRAM を約 0.2GB に保つ仕組みで、置き換える約 2.2GB のアリーナとの差の源泉である。
- **共有バッファとパディング経路**：バッファは最大バケットで一度だけ確保して全バケットで共有する。バケット非整合の T は永続パディングバッファへコピーして流し、末尾の計算結果は出力スライスで捨てる（ゼロ化は不要）。
- **glue の torch.compile**：入口キャストと pixel shuffle は inductor でメモリ帯域の下限に達する。inductor には triton が必要で、Linux は torch 同梱の pytorch-triton、Windows は依存に加えた triton-windows wheel（torch 2.12 に合わせて 3.7 系。Windows 実機で動作確認済み）を使う。triton が無い環境では eager に自動で切り替わり、コンパイルが実行時に失敗した場合も警告を出して eager へ恒久降格する（warmup が両 glue 経路を踏むため、失敗は起動時に顕在化する）。キャストは `dynamic=True` でコンパイルし、バケット形状ごとの再コンパイルを避ける。
- **CUDA graph capture は使わない**：lada-ex で効果がないことを実測済みである。

### 構築の失敗を即時に顕在化させる設計

最大バケットの graph ビルドはコンストラクタ内で即時に行い、warmup の try/except の外に置く。
cuDNN が FP8 カーネルを提供できない環境（古い cuDNN、非対応 GPU）では、ここで例外が上がって呼び出し側が TensorRT へフォールバックする。
ビルドを warmup 内に置くと、例外が握り潰されて登録だけ成功し、実行中のワーカースレッドで初回呼び出しが死ぬ（lada-ex で実際に踏んだ穴である）。

warmup はベストエフォートで、graph ビルド、inductor コンパイル、アロケータのプール成長、ダミー forward 2 種（バケット一致経路とパディング経路）をロード時に前倒しする。
jasna はクリップ長が 1〜`max_clip_size` で自由に分布するため、warmup では**全バケットを事前ビルドする**。
チャンク長が 90/80/70 に固定されていた lada-ex は上位 3 バケットの事前ビルドで足りたが、jasna で同じにすると残りバケットの遅延ビルド（1 回 0.1〜0.2 秒、1 走行あたり計 1 秒前後を実測）が復元ステージの実行中に混入する。
warmup の所要は 1.9 秒（inductor キャッシュが温まった状態。初回はコンパイル込みで 5 秒前後）である。

### nvvfx との cuDNN 共存問題

`--secondary-restoration rtx-super-res` と併用すると、当初は FP8 バックエンドの構築が SIGABRT で落ちた。
原因は二段構えである。

nvvfx（RTX Video Effects）はロード時に、自分の同梱ライブラリディレクトリを `os.environ["LD_LIBRARY_PATH"]` の先頭に追記する。
そこには libcudnn.so.9 のディスパッチャ（9.7 系、約 125KB）だけが置かれ、ディスパッチャが実行時に dlopen する `libcudnn_graph.so.9.7.*` などのサブライブラリは同梱されない。

一方 cudnn-frontend は import 時に `LD_LIBRARY_PATH` を明示的に走査し、最初に見つかった libcudnn.so.9 を絶対パスで CDLL してそのハンドルを使う。
その結果、rtx-super-res を先に構築した後に FP8 バックエンドを構築すると、frontend が nvvfx のサブライブラリ欠落ディスパッチャを掴み、`cudnnCreate` の解決に失敗して abort する。
ELF のロード順の問題ではなく Python レベルのパス探索の問題なので、ライブラリの先行ロード（preload）では直らない。

対策として、`fp8_upsample.py` の import 時に torch 同梱の完全な cuDNN の lib ディレクトリを `LD_LIBRARY_PATH` のさらに先頭へ置く。
frontend の走査が正しいディスパッチャ（9.20）を先に見つけるようになり、nvvfx は自分のコピーを絶対パスでロードし続けるため影響を受けない。
修正後、rtx-super-res 併用の全評価走行が正常終了している。

Windows には別の既知問題（cudnn-frontend の cudart shim が Linux の soname しか探さない）があり、torch 同梱の `cudart64_*.dll` を `CUDNN_FRONTEND_CUDART_LIB_NAME` に設定する回避をモジュール先頭に置いてある（lada-ex の修正の移植）。

## CLI と有効化

```bash
jasna --input in.mp4 --output out.mp4 --fp8-recon
```

`--fp8-recon`（`BooleanOptionalAction`、既定 False）は環境変数 `JASNA_FP8_RECON=1` へブリッジされ、restorer 構築時のゲートがこれを読む。
GUI にトグルはまだ無いが、環境変数を設定して起動すれば同じ経路が有効になる。

デバッグ用の環境変数が二つある。

- `JASNA_FP8_RECON_NOCOMPILE=1`：glue の torch.compile を無効化して eager で動かす（triton の無い環境は元から eager）。
- `JASNA_FP8_RECON_NOWARM=1`：warmup をスキップする（遅延ビルドで動作は継続する）。

有効化の条件は、`JASNA_FP8_RECON=1`、fp16 モード（fp32 エンジン構成は対象外）、CUDA デバイス、sm89 以上、`nvidia-cudnn-frontend` の import 成功、cuDNN の FP8 カーネル提供、のすべてである。
どれか一つでも欠けると TensorRT エンジンにフォールバックする。

## 評価結果

評価は三つの高度で行った。
ステージ単体の A/B（律速に依存しない直接効果）、パイプラインの before/after（per-stage 計時と律速判定）、出力品質と決定性である。

### ステージ単体 A/B

`jasna --benchmark --benchmark-filter fp8` で、同一プロセス・同一の実活性入力（実クリップを split forward に通して採取した `(90, 320, 64, 64)`）に対する TensorRT FP16 エンジンとの比較を行う。
数値は 50 回計測の中央値である。

| T | TensorRT FP16 | FP8 | 速度比 |
|---|---|---|---|
| 10 | 1.38 ms | 0.95 ms | 1.45x |
| 30 | 4.19 ms | 2.79 ms | 1.50x |
| 60 | 8.54 ms | 5.46 ms | 1.56x |
| 90 | 12.82 ms | 8.39 ms | 1.53x |

品質は FP32 の eager 参照に対し、残差加算後の float 領域で per-frame PSNR / SSIM を取った。

| 経路 | PSNR (mean / min) | SSIM |
|---|---|---|
| FP8 | 64.0 dB / 63.6 dB | 0.99976 |
| TensorRT FP16（フロア） | 92.0 dB / 91.4 dB | 1.00000 |

VRAM は driver レベル（`mem_get_info`）の差分で、TensorRT upsample エンジンのロードが **2210 MB**、FP8 バックエンドの常駐が **219 MB**、純減 **1991 MB** である。
FP8 出力は 2 回実行でビット一致し（バケット一致経路・パディング経路とも）、構築と warmup の所要は 1.9 秒（初回のみ inductor コンパイル込みで約 5 秒）である。

lada-ex の参照実測（T=60 で 1.62 倍、PSNR 67.7 dB）と整合する。

### パイプライン before/after

6 本の実クリップ、secondary なしと `--secondary-restoration rtx-super-res` の 2 構成、FP8 on/off の全 24 条件で計測した（test1 と test2 は各条件 2 走行のインターリーブ、他は 1 走行）。
wall-clock は走行全体、VRAM は 200ms 周期の `nvidia-smi` サンプラの peak / steady-state である。

secondary なし：

| クリップ | 内容 | wall (FP16 → FP8) | e2e fps 差 | VRAM peak 差 | VRAM steady 差 |
|---|---|---|---|---|---|
| test1 | 852×480, 10,661f | 61s → 63s | −3.6% | −1405 MB | −1445 MB |
| test2 | 1080p, 31,524f | 197s → 198s | −0.4% | −1556 MB | −1582 MB |
| test3 | 720p, 35,956f | 192s → 193s | −0.7% | −1384 MB | −1610 MB |
| test4 | 720p, 238,380f | 1418s → 1408s | +0.7% | −1592 MB | −1578 MB |
| test5 | 4K, 32,105f | 380s → 382s | −0.5% | −1173 MB | −1268 MB |
| test6 | 4K, 4,754f | 64s → 67s | −4.2% | −1198 MB | −1274 MB |

rtx-super-res 併用：

| クリップ | wall (FP16 → FP8) | e2e fps 差 | VRAM peak 差 | VRAM steady 差 |
|---|---|---|---|---|
| test1 | 71s → 73s | −2.6% | −1513 MB | −1512 MB |
| test2 | 232s → 234s | −0.6% | −1572 MB | −1579 MB |
| test3 | 219s → 220s | −0.7% | −1668 MB | −1598 MB |
| test4 | 1718s → 1715s | +0.2% | −1453 MB | −1664 MB |
| test5 | 406s → 408s | −0.4% | −936 MB | −1330 MB |
| test6 | 70s → 73s | −4.0% | −1190 MB | −1204 MB |

### 律速分析：fps が動かない理由

per-stage 計時（`[timing]` ログ）では、全 12 構成で decode-detect ステージが busy 率 100% の律速であり、primary restore ステージは大きな queue-wait（入力待ち）を抱えている。
producer/consumer パイプラインの e2e fps は律速ステージが決めるため、律速でない restore 内のステージを速くしても fps は動かない。
実測でも長尺クリップの wall-clock 差は ±1% 以内に収まり、これは設計どおりの結果である。

FP8 化の速度利得は `primary.restore` の合計秒にも実質現れない（±1%、最大の test4 で −5.2 秒 / 1044.6 秒）。
upsample はクリップ 1 本の forward あたり数 ms（T=60 で 8.5 ms → 5.5 ms）しか占めず、restore ステージの大半は伝播 4 パスと前処理が占めるためである。
実際、split forward 全体（T=60）の実測は TensorRT 構成で 78〜79 ms、FP8 構成で 75.5 ms であり、削減幅は単体計測の 3.1 ms とほぼ一致する（forward 換算で約 4%）。
言い換えると、このステージ置換の速度面の寄与は「restore ステージの数%を 1.5 倍にする」規模であり、restore が律速になる構成（検出が極端に軽い、または restore 負荷が極端に高い場合）でだけ fps に現れうる。

60 秒級の短尺クリップでは wall-clock が 2〜3 秒増える（fps で −3〜−4%）。
これは FP8 バックエンドの構築と warmup（約 2 秒）が走行長に対して償却されないためで、長尺では消える。

一方 VRAM は全 12 構成で peak −0.9〜−1.7 GB、steady −1.2〜−1.7 GB と一貫して削減された。
削減量はクリップ解像度に依存しない（アリーナは 256² クロップと `max_clip_size` に対するもので、動画解像度と無関係のため）。
高解像度ほどデコードサーフェスやフレームキューが VRAM を圧迫する（本計測でも 4K は FP16 で peak 10.5〜11.0 GB に達する）ので、この固定分の解放は高解像度ほど絶対的な余裕として効く。
16GB GPU では、VRAM 逼迫時に `vram_offloader` が RAM 退避でスループットを落とす前の余裕、あるいは secondary restorer 併用の成立余地がこの約 1.5 GB で変わる。

### 出力品質（パイプライン出力の FP16 対比）

同一入力の FP8 off/on 出力ペア（secondary なし、同一 NVENC 設定）を per-frame PSNR / SSIM で比較した。
この数値は FP16 出力と FP8 出力の「差」であって、正解（FP32）に対する劣化量ではない点に注意する（両者とも FP32 モデル出力の近似であり、ステージ単体の FP32 対比は前掲のとおり 64 dB である）。

| クリップ | PSNR (mean / min) | SSIM (mean / min) |
|---|---|---|
| test1 (480p) | 43.5 dB / 41.1 dB | 0.9825 / 0.9733 |
| test2 (1080p) | 44.6 dB / 41.4 dB | 0.9869 / 0.9773 |
| test3 (720p) | 45.6 dB / 42.8 dB | 0.9890 / 0.9796 |
| test4 (720p) | 45.1 dB / 41.7 dB | 0.9862 / 0.9771 |
| test5 (4K) | 48.7 dB / 45.4 dB | 0.9942 / 0.9880 |
| test6 (4K) | 47.4 dB / 44.4 dB | 0.9932 / 0.9851 |

計画時のゲート（mean 45 dB 以上）に対し、test1 と test2 は 43.5 / 44.6 dB で下回った。
ただし差の実体を確認した結果、品質は同等と判定した。
根拠は三つある。
第一に、最悪フレーム（41.1 dB）でも目視で区別できず、差分マップは復元領域に限定されずフレーム全面に平均 1.5 LSB（最大 22/255）で広がる。これは復元領域の微小差をエンコーダのインター予測が GOP 内へ拡散させた形であり、復元品質の差がそのまま現れたものではない。
第二に、準ロスレス設定（`--encoder-settings cq=1`）で再測しても test1 は 44.7 dB とほぼ変わらず、圧縮の強さではなくエンコーダの経路分岐（モード選択の変化）が支配的である。
第三に、復元領域がフレームに占める割合が小さいほど数値が上がる傾向（4K で 47〜49 dB）とも整合する。
lada-ex の Windows 検証における同種の計測は 49 dB / SSIM 0.993 で、同じオーダーである。

### 決定性

ステージ単体では FP8 出力は 2 回実行でビット一致する。
パイプライン出力の決定性は、まず FP16 同士の 2 走行ペアでフロア（エンコーダやスレッド起因の揺らぎ）を確立し、FP8 同士の 2 走行ペアがフロアと同等以上に一致することを確認する方式を取った。

| ペア | test1 (10,661f) | test2 (31,524f) |
|---|---|---|
| FP16 run0 vs run1（フロア） | md5 一致 | md5 一致 |
| FP8 run0 vs run1 | md5 一致 | md5 一致 |

FP16 パイプラインは走行間で完全決定的（デコード生フレームの md5 一致）であり、FP8 パイプラインも同じく md5 一致した。
FP8 化は決定性を一切損なわない。

## 制限と注意点

- **sm89 / sm90 では速度利得が未検証**である。cuDNN の FP8 カーネル自体は存在するが、1.5 倍の実測は sm120（Blackwell）のみで得ており、Ada / Hopper では TensorRT FP16 エンジンに対する優位が成立しない可能性がある。opt-in と自動フォールバックで安全側に倒してある。
- **Windows でも動作確認済み**である。Windows 11 / 同一 RTX 5080 上で、ステージ単体 A/B の全ゲート通過（速度比 1.42〜1.56 倍、PSNR 64.1 dB / min 63.6、SSIM 0.99976、FP8 常駐 232 MB / 純減 1976 MB、両経路ともビット決定的）と、`--fp8-recon` 付き 1080p フルパイプライン走行の正常完了を確認した。triton-windows wheel 経由の inductor コンパイルも実機で動作する（初回はコンパイル込みで約 20 秒、キャッシュ後は 2〜3 秒。コンパイル失敗時は従来どおり eager glue へ自動降格）。frozen build（PyInstaller / Nuitka）での `import cudnn` / triton と DLL 解決は引き続き未検証の開放事項で、失敗しても TensorRT フォールバックで実走は継続する。
- **起動コストが約 2 秒増える**（初回のみ inductor コンパイル込みで約 5 秒）。60 秒級の短尺クリップでは相対的に目立つ。
- `forward` の戻り値は永続出力バッファのビューであり、次の forward で上書きされる。現在の唯一の呼び出し側は直後の残差加算で消費するため問題ないが、新しい呼び出し側を書く場合は注意を要する。
- fp32 エンジン構成（`--fp16` 無効）は対象外で、FP8 ゲートは成立しない。

## テスト

`tests/test_fp8_upsample.py` に、GPU 不要のゲート・フォールバック・数値検証（mock、pixel shuffle の数学検証、バケット丸め、CLI パース）と、GPU 必須（CUDA + sm89 以上 + cudnn frontend）の FP32 対比パリティ、ビット決定性、`release()` の冪等性テストがある。
`tests/test_basicvsrpp_sub_engines.py` には `load_upsample=False` でロードが 5 エンジンに減ることのテストを追加した。

ステージ単体 A/B はリポジトリ内のベンチマークで再現できる：

```bash
# ステージ単体 A/B（ゲート判定と FP8_AB_JSON 出力）
jasna --benchmark --benchmark-filter fp8

# split forward 内訳（FP8 有効時は upsample 行が FP8 経由になる）
JASNA_FP8_RECON=1 jasna --benchmark --benchmark-filter basicvsrpp
```

パイプライン A/B と出力比較はリポジトリ外のローカル評価ハーネスで行った。
手順は次のとおりで、同等のスクリプトで再現できる：`--fp8-recon` の on/off ペアを同一入力・`--log-level info` で走らせ、wall-clock、stderr の `[timing]` 4 行、`nvidia-smi --query-gpu=memory.used -lms 200` のサンプル（peak / steady-state）を走行ごとに収集する。
出力ペアは av でフレームを揃えてデコードし、per-frame PSNR（全フレーム）、SSIM（サンプリング）、決定性はデコード生フレームの md5 で比較する。
