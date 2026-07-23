# FlashVSR オフライン二次復元(`+modi`)

`--secondary-restoration flashvsr` は、一次復元された 256px のモザイククロップを
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR)(one-step streaming
diffusion VSR。jasna は
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) fork を使用)
で 1024px(4x)へ拡大し、一次の BasicVSR++ では大きなモザイク領域・接写・4K
素材でぼやけがちなテクスチャの写実性を補う。

FlashVSR には 2 つのモードがある:

- **`--secondary-restoration flashvsr`(オフライン 3 段)** — 中間ファイルを介する 3
  プロセス構成。**12 GB 級の GPU** でも動き、パッチ不要の FlashVSR checkout で使える。
  段階再開が可能。以下は主にこのモードの説明。
- **`--secondary-restoration flashvsr-inline`(inline、単一パス)** — 通常のストリー
  ミングパイプラインに FlashVSR を挟む。**中間ファイル・ディスクゲート・二重 encode
  が無い**。**16 GB カード + tiny-long パッチ当ての FlashVSR checkout が前提**。詳細は
  末尾の「inline モード」を参照。

オフライン 3 段が存在する理由: FlashVSR の tiny モードは**単体で 12–16 GB VRAM** を
消費するため、jasna の一次パイプラインと 16 GB カード上で同時常駐できない。ピーク
VRAM が時間的に重ならないよう処理をプロセス分割することで初めて収まる。inline モードは
FlashVSR の **tiny-long**(定メモリ ~11.9 GB、パッチ要)を使い、一次(fp8-recon で
~1.6 GB)と同時常駐させることで単一パスを実現する。

## 仕組み — オフライン 3 段

`--secondary-restoration flashvsr` の 1 コマンドが 3 つのサブプロセスを順に実行する。
各段は次段が始まる前に完了し、プロセス終了時に VRAM を全解放するため、ピーク VRAM が
同時に存在することはない:

| 段 | 環境 | ~VRAM | 内容 |
|----|------|-------|------|
| 1 (dump) | jasna | ~9 GB | decode + detect + BasicVSR++ 一次復元。各 clip の 256px クロップ + マスク + 幾何をディスク上の **bundle** へ直列化。blend/encode は捨てる。 |
| 2 (FlashVSR 4x) | FlashVSR | 12–16 GB | 各 clip の 256px クロップを 1024px に拡大し bundle へ書き戻す。 |
| 3 (reblend) | jasna | 軽い | source を再デコードし、bundle から復元結果を再構成、1024px クロップを再 blend して最終出力を encode。 |

Phase 1 / Phase 3 は `jasna --flashvsr-phase {dump,reblend}` のサブプロセスとして
走る(`jasna/__main__.py` で multiprocessing ガードより前に分岐。`--compile-engines`
と同じ流儀)。Phase 2 は `jasna/restorer/flashvsr_phase2_driver.py` を FlashVSR
仮想環境の Python で実行する(jasna を import しない独立スクリプト)。

**bundle** は numpy/JSON ファイルのディレクトリ(`manifest.json`、clip ごとの
`clip_<track>_<start>.npz` と Phase 2 が書く `_fvsr.npz`)。`--flashvsr-bundle-dir`
を指定すると永続化され、途中で失敗した実行を失敗した段から再開できる(完了済み
clip はスキップ)。

blend に必要な幾何(`scale_offsets`)は blend 時に復元フレームの実寸から導出される
ので、FlashVSR の 4x 出力は**メタデータ改変ゼロ**で再 blend できる。

## 必要なもの

FlashVSR は**同梱していない**。
[`lihaoyun6/FlashVSR_plus`](https://github.com/lihaoyun6/FlashVSR_plus) fork の
checkout・重み・専用仮想環境を利用者が用意し、`--flashvsr-repo` で jasna に渡す。

### FlashVSR checkout のセットアップ(一度だけ)

RTX 5080(sm120, 16 GB)/ Linux / CUDA 13.0 で検証済みの再現手順。torch
2.13.0+cu130 / triton 3.7.1 になる:

```bash
# 1. jasna が対象とする fork を clone。models/posi_prompt.pth もこれで入る
#    (repo に git-track されており、ダウンロードではない)。
git clone https://github.com/lihaoyun6/FlashVSR_plus
cd FlashVSR_plus

# 2. uv-managed の *standalone* Python venv を作る。これは必須。FlashVSR の Triton
#    Sparse_SageAttention カーネルは実行時 JIT され Python 開発ヘッダ(Python.h)を
#    要するが、system / conda の Python には同梱されず JIT が
#    「fatal error: Python.h」で落ちる。uv の managed Python はヘッダを含む。
uv venv --python 3.13 --python-preference only-managed

# 3. CUDA に合う wheel index で FlashVSR の依存を venv に入れる
#    (jasna は cu130 で検証。CUDA 12.8 なら .../whl/cu128)。
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu130

# 4. 重み(~6.5 GB)は models/FlashVSR-v1.1/ に置かれる。初回実行時に HuggingFace から
#    自動ダウンロードされるので本手順は任意。jasna の Phase 2 中にダウンロードしたく
#    なければ先に取得しておく:
.venv/bin/huggingface-cli download JunhaoZhuang/FlashVSR-v1.1 --local-dir models/FlashVSR-v1.1

# 5. (推奨)jasna に組み込む前に FlashVSR 環境単体でスモークテスト。jasna の Phase 2
#    が使う tiny / sage / bf16 の 4x パスそのものを叩き、手順4を省いた場合は重み
#    ダウンロードも走る:
.venv/bin/python run.py -i ./inputs/example0.mp4 -s 4 -v 11 -m tiny -d cuda:0 -t bf16 -a sage ./_smoke
```

補足:
- `sageattention` pip パッケージは**不要**。`-a sage` が使うのは fork が同梱する
  `sparse_sage` カーネルで、`sageattention` の import は guard 済み。
- 完了後 `<repo>/models/FlashVSR-v1.1/` に
  `diffusion_pytorch_model_streaming_dmd.safetensors`・`Wan2.1_VAE.pth`・
  `LQ_proj_in.ckpt`・`TCDecoder.ckpt`、隣に `<repo>/models/posi_prompt.pth` が揃う
  ——これが `--flashvsr-repo` の期待する構成。

### jasna から指定するもの

- `--flashvsr-repo <path>`(必須): 上で作った `FlashVSR_plus` checkout。
- `--flashvsr-python <path>`(既定 `<repo>/.venv/bin/python`): 手順2の uv-managed
  venv の Python。
- `--flashvsr-model-dir <path>`(既定 `<repo>/models/FlashVSR-v1.1`): 重み。

## 使い方

```bash
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration flashvsr \
      --flashvsr-repo ~/FlashVSR_plus \
      --log-level info
```

### フラグ

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--flashvsr-repo` | (必須) | `FlashVSR_plus` checkout のパス。 |
| `--flashvsr-python` | `<repo>/.venv/bin/python` | FlashVSR 環境の Python(uv-managed standalone venv)。 |
| `--flashvsr-model-dir` | `<repo>/models/FlashVSR-v1.1` | FlashVSR 重みディレクトリ。 |
| `--flashvsr-version` | `11` | モデル版(`10` / `11`)。 |
| `--flashvsr-dtype` | `bf16` | 計算 dtype(`fp16` / `bf16`)。 |
| `--flashvsr-max-clip-frames` | `32` | Phase 1 の `--max-clip-size` を上限化し、各 clip を FlashVSR tiny の VRAM に収める。 |
| `--flashvsr-unload-dit` / `--no-flashvsr-unload-dit` | on | VAE decode 前に DiT をオフロード(VRAM 節約)。 |
| `--flashvsr-tiled-vae` / `--no-flashvsr-tiled-vae` | on | FlashVSR の VAE decode をタイル化(VRAM 節約)。 |
| `--flashvsr-tiles` | `1` | inline 専用: DiT 推論を横短冊に分割して VRAM ピークを下げる(`2`〜`4`)。オフラインは無視する。詳細は「[strip タイリング](#strip-タイリング--flashvsr-tiles)」。 |
| `--flashvsr-bundle-dir` | temp | 中間 bundle をここに永続化(段階再開が可能に)。 |
| `--flashvsr-keep-bundle` | off | 完了後も bundle を残す(`--flashvsr-bundle-dir` 指定時は暗黙的に有効)。 |

FlashVSR は 4x 固定。`--flashvsr-scale` は無い。

### clip 長を上限化する理由

Phase 2 は FlashVSR の **tiny** モードを使う。tiny は全 latent フレームを VRAM に
保持し、ロスレスな tensor を返す。パイロットでは 16 GB カードで 21 フレーム=~13.5 GB、
65 フレームで near-OOM だった。そのため一次の clip 長を上限化し
(`--flashvsr-max-clip-frames`、既定 32)、各 clip をその予算内に収める。これが
FlashVSR モードで clip が通常より短くなる理由で、増える継ぎ目は clip 境界の crossfade
が吸収する。上限を上げると Phase 2 で OOM の恐れがある。

## ディスク容量

bundle の容量は Phase 2 の**非圧縮 1024px 出力**が支配する。復元クロップ 1 枚が
1024×1024×3 ≈ **3 MiB** で、256px 一次 dump は 1 clip 丸ごとで ~3 MiB。つまり
bundle 容量はモザイク・クロップ枚数に比例し、動画が長いほど増える:

- 目安: **モザイクを含む 1 ソースフレームあたり ~4 MB** ≒ 全編モザイクの 30fps 動画で
  **1 分あたり ~8 GB**(モザイクが一部の時間帯だけなら比例して少ない)。
- 実測: 6 分 / 10,661 フレーム・全編モザイク・510 clip → **~46 GB**
  (1024px 出力 ~45 GB + 256px dump ~1.6 GB)。
- 長尺・モザイク多めの動画では **数百 GB** に達しうる。

ピークは **bundle 全量**。Phase 2 が全 clip の 1024px を書き終えてから Phase 3 が
始まるため、全 fvsr が同時にディスク上に存在する。

> ⚠️ **既定の bundle はシステム temp(`/tmp`)下に作られる。Linux では `/tmp` が
> しばしば `tmpfs`(RAM 上)で数十 GB しかない。**大きな bundle をそこに書くと
> `/tmp` が溢れ RAM を食い潰して失敗する。短いクリップ以外では
> `--flashvsr-bundle-dir <path>` で bundle 全量が入る実ディスクを指すこと
> ——目安は *モザイク分数 × 8 GB*。段階再開も可能になる。

jasna は自動でこれを見張る: Phase 1 前に bundle dir が tmpfs なら警告し、空き容量と
最悪ケース見積りを表示。さらに Phase 1 後(実 clip 数が判明後)に 1024px 出力の正確な
サイズを計算し、**入り切らなければ高コストな Phase 2 を始める前に中断する**(bundle は
保持されるので `--flashvsr-bundle-dir` を大きいディスクに向けて再開できる)。

## 制約

- **ファイル出力専用**。`--stream`・フォルダ/画像入力・`--frame-gen` とは併用不可
  (フレーム生成は出力に対する別パスとして実行する)。
- **fps リターゲット・スマートレンダリング・VR 非対応**。`--retarget-high-fps`
  (Phase 1 の frame stride が Phase 3 の再ブレンド索引とずれる)・`--segments`・
  VR 処理(`--vr-mode sbs`/`sbs-fisheye`、または `auto` が VR コンテンツを検出した
  場合 — Phase 3 に VR プロジェクタがない)は起動時に拒否する。
- **encode が 2 回**。Phase 1 は完全にテスト済みのパイプラインをそのまま流すため
  捨て出力を encode し、最終 encode は Phase 3 で行う。通常実行より encode が 1 回多い。
- **同梱なし / サポーターモデルとは無関係**。FlashVSR は独自ライセンスのサードパーティ
  モデル。checkout・重み・venv は利用者が用意する。jasna のサポーターモデルとは無関係。

## inline モード(`--secondary-restoration flashvsr-inline`)

オフライン 3 段と同じ FlashVSR checkout / 重み / venv・同じ `--flashvsr-*` フラグ
(`repo` / `python` / `model-dir` / `version` / `dtype`)を使うが、**中間ファイルを
一切作らず**、jasna の通常のストリーミングパイプラインの中で FlashVSR を二次復元として
走らせる。

```bash
jasna --input in.mp4 --output out.mkv \
      --secondary-restoration flashvsr-inline \
      --flashvsr-repo ~/FlashVSR_plus \
      --log-level info
```

### オフラインとの違い

| | `flashvsr`(オフライン 3 段) | `flashvsr-inline` |
|---|---|---|
| パス | dump → FlashVSR → reblend の 3 プロセス | 単一ストリーミングパス |
| 中間ファイル | 256px + 1024px bundle(数十 GB 級) | **無し** |
| encode 回数 | 2(捨て + 最終) | 1 |
| FlashVSR モード | tiny(O(T)、~12–16 GB) | **tiny-long(O(1)、~11.9 GB)** |
| 必要 VRAM | 各段が非同時なので実質 tiny 単体分 | primary と**同時常駐**(実測 ~14.8 GB @16 GB カード) |
| FlashVSR checkout | パッチ不要 | **tiny-long マルチチャンク修正のパッチ必須** |
| 段階再開 | 可(bundle 永続化) | 不可(単一パス) |
| 進捗 / キャンセル / GUI | 3 段フロー | 通常 secondary と同じ |

### 前提: tiny-long パッチ

inline は VRAM 定常(O(1))の **tiny-long** を使う。FlashVSR_plus の tiny-long は
第 2 チャンクで壊れる既知バグ(`8192 vs 4096` エラー)があり、**修正パッチを当てた
checkout が必須**。jasna は起動時に checkout を検査し、未パッチなら明示エラーで停止して
`flashvsr`(オフライン、tiny、パッチ不要)を案内する。

パッチ本体は
[`patches/flashvsr_plus_tinylong_multichunk_fix.patch`](../../patches/flashvsr_plus_tinylong_multichunk_fix.patch)
に同梱。FlashVSR_plus checkout で当てる:

```bash
cd ~/FlashVSR_plus
git apply /path/to/jasna/patches/flashvsr_plus_tinylong_multichunk_fix.patch
```

やっていることは 2 箇所のチャンク跨ぎキャッシュ clear を無効化するだけ
(`src/pipelines/flashvsr_tiny_long.py` の per-chunk `LQ_proj_in.clear_cache()` と
`TCDecoder.clean_mem()` を削除。ループ前の一度きりのリセットは残す)。

### 挙動と制約

- **clip 32 上限・frame-gen off を強制**(オフラインと同じ理由)。`--max-clip-size` は
  自動的に 32 へ丸められる。
- **fp8-recon を自動有効化**(未指定時)。一次のピークを ~0.9–1.7 GB 下げ、同時常駐の
  予算に収める。GPU が fp8 非対応(sm89 未満 / `--fp16` 無し)なら TRT へフォールバック。
- 同期実行。FlashVSR(~15 crop-fps)が律速なので、モザイクが多い区間はその速度に
  律速される(モザイクの無いフレームは一次のみで高速)。FlashVSR が壁時計を支配する
  ため、`--batch-size` を下げても速度低下はほぼ無い。
- VRAM(**16 GB カード + デスクトップ常駐**時): 480p で combined ~14.8 GB。ただし
  **1080p 以上は物理天井際**まで上がる(実測 ~15.8 GB ピーク)。worker の
  `expandable_segments` と jasna の `vram_offloader`(キューフレームを system RAM へ
  退避)が圧を吸収して落ちない(1080p では
  `expandable_segments: memory mapping failed with OOM` の**警告**(無害。クラッシュ
  ではない)と大量の offload が出る)。天井が近いときの第一の対策は
  **`--flashvsr-tiles`**(次節)。補助として `--batch-size 2`(または `1`)や
  MPS 停止(~490 MB 増)もある。VRAM が少ない環境や未パッチ checkout では
  オフライン(`flashvsr`)を使う。
- **Windows では `expandable_segments` が使えず worker の reserved が ~13 GB に膨らむ**
  ため、tiles 無しの inline は物理天井に張り付く(完走はするが余裕がほぼ無い)。
  1080p では **`--flashvsr-tiles 2` を推奨**。実測は
  「[Windows での注意事項](#windows-での注意事項)」。

### strip タイリング(`--flashvsr-tiles`)

inline 専用の VRAM 対策。各 256px クロップを幅そのままに高さ方向だけ横短冊(strip)に
分割し、短冊ごとに tiny-long を回して羽根(feather)合成する。DiT のトークン活性メモリ
(特に block-sparse draft の attn マスク)はタイル面積の二乗で減るため、少ない計算増で
ピーク VRAM が下がる。オフライン(`flashvsr`)は本フラグを無視する。

| `--flashvsr-tiles` | 短冊 | attn マスク(対 full) | 計算量(対 full) |
|---|---|---|---|
| `1`(既定) | なし(単発) | 1.0 | 1.0 |
| `2` | 2 枚(各 256w×160h) | 0.39x | ~1.25x |
| `3` | 3 枚(各 256w×128h) | 0.25x | ~1.5x |
| `4` | 4 枚(各 256w×96h) | 0.14x | ~1.5x |

短冊は少ないほど速く品質も良い(重複計算が少なく、1 短冊の空間文脈が広い)ので、
VRAM が許す**最小の枚数**を選ぶ。2 で収まれば 2、天井に張り付く/OOM するなら 3、
それでも足りなければ 4。

品質への影響: 短冊境界は羽根合成され、実機確認(Windows / RTX 5080、tiles 1 との
同一フレーム比較)ではバンディング・段差・短冊間の色調ずれは検出されなかった。差分は
拡散モデルの確率的なテクスチャ揺らぎの範囲にとどまる。なお合成の都合上、出力の最外
1px は重み 0 になる(本家 run.py 由来の既存挙動。ブレンド時のクロップ境界は羽根が
かかるため実害は小さい)。

### 実装

- 同期 `SecondaryRestorer`: `jasna/restorer/flashvsr_inline_secondary_restorer.py`
  (FlashVSR venv worker を resident spawn、length-prefixed の RGB wire、
  `close()` で終了)。
- worker(FlashVSR venv 実行、jasna 非依存): `jasna/restorer/flashvsr_inline_worker.py`
  (tiny-long pipe、`imageio.get_writer` を差し替えてロスレスにテンソル捕獲、
  small clip は next_8n5 パディングで吸収し厳密に T 枚返す。strip の分割と
  羽根合成もここ)。
- CLI 配線: `jasna/main.py`。テスト: `tests/test_flashvsr_inline.py`。

## Windows での注意事項

検証環境: Windows 11 / RTX 5080 16 GB / torch 2.13.0+cu130(FlashVSR venv)。結論:
**16 GB カードでは、オフライン(`flashvsr`)か、inline + `--flashvsr-tiles`
(1080p は `2` 推奨)を使う。tiles 無しの inline は完走はするが余裕がほぼ無い。**

- **PyTorch の `expandable_segments` は Windows 未対応**(警告を出して既定の
  キャッシングアロケータへフォールバック)。tiny-long の reserved VRAM は Linux の
  「フラット ~11.9 GB」より断片化で **+1〜2 GB** 膨らむ。`backend:cudaMallocAsync`
  でも改善しない(実測でむしろ微増)。jasna は Windows では worker に
  `expandable_segments` を設定しない。
- **WDDM デスクトップ常駐が ~1 GB** を取る(ヘッドレス Linux ではほぼ 0)。16 GB
  カードの実効空きは **~15.2 GB**。ブラウザや IDE も開いた実デスクトップでは
  アイドルで ~2 GB を超えることもある。
- inline のフルパイプライン実測(フル長素材、nvidia-smi の GPU 全体ピーク、
  アイドル ~2.1 GB の実デスクトップ常駐):

  | `--flashvsr-tiles` | 1080p ピーク | 壁時計(対 tiles 1) |
  |---|---|---|
  | `1` | 15918 MiB | 1.00x |
  | `2` | **14222 MiB** | 1.25x |
  | `3` | 12490 MiB | 1.46x |
  | `4` | 11514 MiB | 1.46x |

  tiles `1` は 480p でも 1080p でも完走した(offload 0 回、OOM 警告 0 件)が、
  ピークは物理天井(16303 MiB)まで 400 MiB を切り、常駐アプリの変動で OOM に
  転じうる。**1080p の常用は `--flashvsr-tiles 2`**(余裕 ~2 GB、減速 +25%)。
  短冊境界のシーム(バンディング、色調ずれ)はこの実測でも検出されなかった。
- 実測ピーク(scale 4 / tiny-long / bf16 / sage / 85 フレーム、reserved 値):
  - **256px 入力(jasna の実ワークロード): ~13.0 GB** — Phase 2 は GPU を単独占有
    するので、オフラインは 16 GB Windows で動く。
  - **384px 入力(同梱 example0 での smoke): ~15.1 GB** — 空きと紙一重。ブラウザや
    IDE が数百 MB 使っているだけで OOM する。**smoke の OOM ≠ jasna 実負荷の OOM**。
- **`-m tiny`(O(T))での 85 フレーム smoke は 16 GB Windows では OOM して正常**。
  smoke は `-m tiny-long` で行う(セットアップ手順 5 のコマンドの `-m tiny` を
  読み替える)。
- venv の Python は `<repo>/.venv/Scripts/python.exe`(`--flashvsr-python` の既定も
  Windows ではこのパスに解決される)。
- stdout がパイプに向く(リダイレクト / 一部の GUI 起動)と、FlashVSR の起動バナー
  (ブロック文字)が cp932 で `UnicodeEncodeError` になり推論前に落ちる。jasna からの
  起動(オフライン Phase 2 / inline worker)は `PYTHONUTF8=1` を自動設定するので
  対処不要。**run.py を手で叩いて出力をリダイレクトする場合は
  `$env:PYTHONUTF8=1` を先に設定**する。
- run.py は出力ディレクトリを自動作成しない(存在しないと推論後の書き出しで
  `FileNotFoundError`)。事前に `mkdir` しておく。

## 実装(オフライン)

- オーケストレータ・bundle 形式・Phase 1 dump hook・Phase 3 reblend:
  `jasna/restorer/flashvsr_offline.py`。
- Phase 2 driver(FlashVSR venv): `jasna/restorer/flashvsr_phase2_driver.py`。
- サブプロセス分岐: `jasna/__main__.py`(`--flashvsr-phase`)。
- CLI 配線 / 早期分岐: `jasna/main.py`。
- テスト: `tests/test_flashvsr_offline.py`、`tests/test_main.py`。

再利用した jasna 資産: `BlendBuffer` / `crop_buffer.scale_offsets`(1024px クロップは
無改変で再 blend)、`RestorationPipeline.build_secondary_result`
(`[keep_start:keep_end]` スライス)、`pipeline_items`(直列化単位)、Phase 3 の
decode/encode に `media/backend.make_video_{reader,encoder}`。
