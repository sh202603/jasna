# LTX 出力の鮮明化 `--ltx-crop-vsr`（`+modi`）

`--ltx-crop-vsr` は、公式リリース版で LTX 復元した動画を入力にして、モザイクがあった領域だけを SwiftVR で ×2 して描き直し、LTX 出力へ合成し直す後処理モードである。
出力の解像度、フレーム数、タイムスタンプは LTX 出力のまま変わらない。

この modi ビルドは公開ソースから作るので、LTX モデルの重み（`.enc`）を復号できず、LTX 復元そのものは走らせられない。
一方で LTX の scan（モザイク追跡と camera、window の計画）と compose（feather 付き合成）のコードは公開されているので、LTX 復元は公式リリースで行い、その出力を modi で鮮明化する分担が取れる。
鮮明化が効くのは、LTX が 512 px（`--ltx-large-canvas` なら 768 px）の canvas で描いた結果を crop サイズへ bicubic 拡大して戻すときのぼけである。
ユーザー目視（720x480 素材）では、くっきり度は SwiftVR > distill ≒ CAS > LTX そのままの順で、揺れは気にならなかった。

## 使い方

```
jasna --input <LTX出力> --output <出力> \
      --ltx-crop-vsr swiftvr --ltx-crop-vsr-source <元動画> \
      --swiftvr-repo <SwiftVR checkout> --swiftvr-model-dir <checkpoint> --cq 14
```

| フラグ | 意味 |
| --- | --- |
| `--input` | リリース版が LTX で復元した動画。書き出すフレーム、音声、字幕、章はここから取る。 |
| `--ltx-crop-vsr-source` | その元動画（モザイクあり）。モザイクの検出と計画、および「LTX が触っていない window」の判定に使う。LTX 出力にはモザイクが残っていないので必須。 |
| `--ltx-crop-vsr swiftvr` | 本家 SwiftVR を `swiftvr-inline` の worker で scale 2 固定で動かす。`--swiftvr-repo`、`--swiftvr-python`、`--swiftvr-model-dir`、`--swiftvr-accel` を使う（[swiftvr.md](swiftvr.md)）。 |
| `--ltx-crop-vsr swiftvr-distill` | distill モデルを jasna のプロセス内で動かす（×2）。`--swiftvr-distill-model`、`--swiftvr-distill-strength`、`--swiftvr-distill-stabilize` を使う。SwiftVR より軽く、質感の上乗せは大きいが揺れも大きい。 |
| `--ltx-large-canvas` | LTX 実行時に 768 px canvas で走った場合だけ付ける。refine では空き VRAM を見ずに 768 を使う。 |

検出モデル（`--detection-model`）は LTX 実行時と同じものを推奨する。閾値と `--batch-size` は同じでなくてもよい。
refine の camera は元動画から独立に計算され、SR 入力はその crop で LTX 出力を切り出したものなので、crop や polygon が LTX 実行時と多少違っても、鮮明化の範囲が少し広がるか狭まるだけで害は無い。
`--ltx-auto-split` や `--segments` で作られた LTX 出力も入力にできる。

### LTX 側の要件

- LTX 出力は高品質（低 CQ）で書き出す。refine はマスク外も再エンコードするため、refine 側も `--cq` を低くする。
- `--lut` と `--sharpen` は LTX 側で使う。refine 側では拒否する（両方で使うと二重適用になる）。
- `--ltx-large-canvas` を付けて走らせた場合、ログに「restoring large mosaics at 512 px」の警告が出ていなければ refine にも付ける。
- 出力コーデックは hevc（10-bit）を推奨する。refine 側の `--codec` 既定も hevc で 10-bit なので、素通し領域のビット深度が保たれる。

### 実行例

作業は 2 パスである。公式リリースで LTX 復元し、その出力をこのビルドで鮮明化する。
refine はマスク外も含めて画面全体をもう一度エンコードするので、両パスに同じ低い `--cq` と同じエンコーダー設定を与え、モザイク以外の部分が途中で劣化しないようにする。

```bash
# パス 1（公式リリース）: LTX 復元を高品質で書き出す。
jasna --input source.mp4 --output ltx.mkv \
      --restoration-model-name ltx --detection-model rfdetr-v6 \
      --codec hevc --cq 14 --encoder-settings "preset=p7,rc-lookahead=32,qmax=28"

# パス 2（このビルド）: LTX 出力のモザイク領域を SwiftVR で鮮明化する。
jasna --input ltx.mkv --output ltx_sharp.mkv \
      --ltx-crop-vsr swiftvr --ltx-crop-vsr-source source.mp4 \
      --detection-model rfdetr-v6 \
      --swiftvr-repo ~/SwiftVR --swiftvr-model-dir ~/SwiftVR/checkpoints \
      --codec hevc --cq 14 --encoder-settings "preset=p7,rc-lookahead=32,qmax=28"
```

SwiftVR の代わりに distill モデルを使う場合のパス 2（checkout も venv も不要。`model_weights/` に `swiftvr-distill.pt` を置く）:

```bash
jasna --input ltx.mkv --output ltx_sharp.mkv \
      --ltx-crop-vsr swiftvr-distill --ltx-crop-vsr-source source.mp4 \
      --detection-model rfdetr-v6 --swiftvr-distill-strength 0.75 \
      --codec hevc --cq 14 --encoder-settings "preset=p7,rc-lookahead=32,qmax=28"
```

エンコード関連オプションの意味:

- `--cq 14` は HEVC の既定 28 よりかなり低い（低いほど高品質で大きいファイル。値はそのまま NVENC に渡る）。モザイク以外の画質はパス 1 で決まり、パス 2 はそれを保つだけでよい。低ビットレートの元動画では、パス 1 の極端に低い CQ が自動の出力サイズ上限で頭打ちになることがある（[エンコーダー設定](cli.md#エンコーダー設定)を参照）。パス 2 の入力は高ビットレートの LTX 出力なので影響しない。
- `--encoder-settings "preset=p7,rc-lookahead=32,qmax=28"`: `preset=p7` は NVENC の最も遅く最も高品質なプリセット（既定 `p5`）、`rc-lookahead=32` はレート制御に 32 フレーム先まで見せる設定、`qmax=28` は VBR が落ちてよい最悪の量子化値の上限（既定 34）。[NVENC のキー一覧](cli.md#nvidianvencのキー全コーデック共通)のどのキーも使え、未知のキーは受理可能なキー一覧を出して停止する。
- `--codec hevc` は両パスの既定で、素通し領域を 10-bit のまま保つ。LTX 出力を 8-bit の `h264` で書いた場合は refine 側にも `--codec h264` を渡し、ビット深度を変えないようにする。
- `--ltx-fast`、`--restoration-model-name ltx-undistilled`、`--ltx-seed` はパス 1 だけのもので、パス 2 には影響しない。`--ltx-large-canvas` は両パスに付ける（上記）。`--lut` と `--sharpen` はパス 1 だけに付ける。

### 拒否される組み合わせ

`--restoration-model-name` が basicvsrpp 以外、`--secondary-restoration`、`--denoise`、`--segments`、`--ltx-auto-split`、`--stream`、`--retarget-high-fps`、`--fmp4`、`--frame-gen`、VR（`--vr-mode` が off 以外に解決された場合）、画像入力、フォルダ入力、`--lut`、`--sharpen`、`--swiftvr-scale 4`、`--swiftvr-view-window` と `--swiftvr-distill-view-window` の既定以外、`--ltx-large-canvas` と `--swiftvr-distill-stabilize` の併用。
GUI は対象外（CLI のみ、単一ファイルのみ）。

## 仕組み

実装は `jasna/ltx/refine.py`。BasicVSR++ は読み込まず、エンジンのコンパイルも検出器の分だけ行う。

1. **整合確認**: 2 本をパケット単位で demux し（デコード無し、数秒）、枚数と解像度が一致し、各フレームの時刻 (pts − start_pts) × time_base が半フレーム以内で一致することを確かめる。生の pts は比べない。jasna のエンコーダは元動画の start_pts を引くし、MKV なら time_base も変わるためである。
2. **scan**: 元動画に対して upstream の `scan_span` を走らせ、LTX と同じ track、camera、window を得る。`LTX plan: N frames, C scene cuts, T tracks, W windows` の info ログはリリース版のログと照合できる。scan が終わると検出器を解放し、それから SwiftVR を起動する。
3. **鮮明化と合成（1 パス）**: LTX 出力と元動画を 2 本の NVDEC で並走デコードする。各 window について、フレームが届くたびに LTX 出力から crop を切り出し、canvas サイズへ縮小（pixel area）または拡大（bicubic）して reflect pad する。window の全フレームが揃ったら ×2 し、結果を `CropFrame` を 2 倍した版と組にして upstream の `composite_frame` で LTX 出力フレームへ合成する。feather マスク、window 間 crossfade、track 間 handover、マスク外の素通しは LTX 本体と同じになる。色合わせ（リング中央値差 × 0.75）は SwiftVR の色ずれを LTX 出力へ寄せる向きに効く。
4. **LTX が触っていない window の skip**: 並走している元動画との輝度差を track polygon 内で平均し、閾値（3/255）未満の window は SR も合成もせず warning を出す。リリースと modi の検出エンジンの数値差で refine 側だけに track が出た場合（LTX 出力にモザイクが残ったまま鮮明化される）と、`--ltx-crop-vsr-source` に同じ枚数の別カットを渡した事故を拾う。逆方向（LTX にあって refine に無い track）はその区間が LTX のまま残るだけで無害。
5. **エンコード**: 既存の NVENC 経路。writer には LTX 出力の pts をそのまま渡すので、出力の pts 列は LTX 出力と同じになる。

### メモリ

- フレームの保持: window の SR が終わるまでそのフレームは書き出せない。保持枚数の上限は track 数によらず 121 枚（frame f を覆う window の終端はどれも f + 121 以下）。CPU pinned、uint8 で、1080p で約 0.75 GB、4K で約 3 GB。
- SR 結果: crossfade のため track ごとに直前の window も保持する。1 window は 1024² で約 380 MB、1536² で約 860 MB（CPU）。
- VRAM: SwiftVR worker（FP8、1024 処理）、デコード 2 本、エンコード。distill は stabilize 0（既定）なら 1 GB 未満の上乗せ、stabilize > 0 では window 全体を float で持つため 512 canvas で約 4.5 GB、768 canvas では約 10 GB になるので併用を拒否する。

## 検証

Linux、RTX 5080 16 GB、2026-10-16。素材はリリース v0.11.1 の LTX 出力とその元動画 2 組（720x480 で 890 枚、1080p で 899 枚、各 30 秒、LTX 出力は CQ 14）。

- **skip 閾値**: LTX が復元した window の輝度差平均は 4.6〜10.4、同じ crop のマスク外（再エンコード雑音）は 1.0 未満。閾値 3.0 はその間に置いた。2 組とも 11 window すべてが復元済みと判定され、skip は無し。
- **完走と整合**: 2 組 × 2 方式が完走し、scan の plan はリリースのログと同じ（2 track、11 window、シーンカット 1）。出力の枚数は LTX 出力と一致し、各フレームの時刻の差は最大 0.5 ms（MKV の time_base 1/1000 の丸め）。元動画の start_pts が 0 でない組も通った。
- **指標**（mioh-labs の texture と flicker。基準は同じ経路での再エンコード）: distill 0.75 は texture 119.9% / 121.7%、揺れの増加 +11.2% / +8.4%（720x480 / 1080p）で、プローブ（118.8% / 118.4%、+14.2% / +9.4%）の許容幅の内。SwiftVR は強度 1.0 で 114.6% / 120.9%、+8.5% / +10.1%。プローブの SwiftVR 行（108.7% / 110.5%）は bicubic 拡大との 0.75 混合だったので直接は比べられず、720x480 でプローブを強度 1.0 にすると 112.8% で 2 ポイント以内。LTX の色合わせを切ると distill の揺れが 2〜3 ポイント増える（SwiftVR は内蔵の wavelet 色補正があり不変）ので、色合わせはそのまま使う。
- **時間と VRAM**（30 秒素材、worker 起動込み）: distill は壁時計 23〜28 秒、VRAM の上乗せ 2.5〜2.8 GB。SwiftVR は 50〜55 秒、上乗せ 9.8〜10.8 GB（設計時の見積り 7 GB より大きい。16 GB カード向けで、12 GB 級では distill を使う）。scan は 3 秒。
- 未了: ユーザー目視、768 px canvas、crop が canvas の半分未満の素材、10 分超の長尺、Windows。

## 制限

- 倍率 crop/canvas が 0.5 未満の track（1080p や 4K の小さいモザイク）は未検証。小さい crop を canvas へ大きく拡大してから SR するので、挙動が変わりうる。
- SR はデコードループ内で同期実行で、デコード、合成、エンコードと重ならない。
- `--retarget-high-fps` で作られた LTX 出力は枚数違いで弾かれる。

## ライセンス

LTX 出力の後処理は LTX-2.x Community License §5（Output）の範囲で、`.enc` には触れない。LTX の入手方法や使い方は upstream のドキュメントを参照。
