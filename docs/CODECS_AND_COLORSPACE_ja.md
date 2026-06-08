# 出力コーデック・ビット深度・色空間（AV1 / 8bit NV12 / BT.601・BT.2020）

`v0.7.1+modi` で追加された出力フォーマットの柔軟化についてのガイドです。

## 概要

jasna の処理経路は元々 **NVDEC → 復元 → NVENC** を GPU 上で完結させる「ゼロコピー」設計で、
CPU への往復はありません（`tests/test_no_cpu_tensor_ops.py` で保証）。本リリースでは、その出力段を
次のように拡張しました。従来は **HEVC / 10bit(P010) / BT.709 固定**でした。

| 項目 | 従来 | 本リリース |
|---|---|---|
| コーデック | HEVC のみ | **HEVC / AV1** |
| ビット深度 | 常に 10bit (P010) | **ソース連動（8bit NV12 / 10bit P010）または明示指定** |
| 色空間 | BT.709 へ強制変換 | **BT.601 / BT.709 / BT.2020 をソースから保持** |

## CLI オプション

### `--codec {hevc,av1}`
出力コーデック。既定 `hevc`。`av1` はファイル出力時のみ対応（後述）。

### `--bit-depth {auto,8,10}`
出力ビット深度。既定 `auto`。
- `auto` — ソースが 10bit なら P010(10bit)、それ以外は NV12(8bit) で出力。
- `8` — 常に 8bit (NV12)。
- `10` — 常に 10bit (P010)。

> **補足:** 復元パイプラインは内部的に常に 8bit RGB で動作します。そのため 8bit ソースを
> 10bit で出力しても情報量は増えず、コンテナが広がるだけです。`auto` を推奨します。

### 使用例

```bash
# 8bit ソース → 8bit HEVC（auto）
jasna --input in_8bit.mp4 --output out.mkv

# AV1 出力（ビット深度はソース連動）
jasna --input in.mp4 --codec av1 --output out_av1.mkv

# 10bit ソースを敢えて 8bit HEVC で出力
jasna --input in_10bit.mp4 --bit-depth 8 --output out_8bit.mkv

# BT.2020 ソース → 色空間を保持したまま AV1 10bit
jasna --input in_bt2020.mp4 --codec av1 --bit-depth 10 --output out.mkv
```

## 色空間の扱い

入力の色空間（ffprobe の `color_space`）を判定し、対応する limited-range の RGB→YUV 係数で
エンコードします。出力コンテナにも色空間メタデータ（マトリクス＋原色＋トランスファ）を付与します。

| ソース color_space | 使用マトリクス | 出力メタデータ（colorspace / primaries / transfer） |
|---|---|---|
| `bt709`（既定）ほか | BT.709 | `bt709` / `bt709` / `bt709` |
| `smpte170m` / `bt601` / `bt470bg` | BT.601 | `smpte170m` / `smpte170m` / `smpte170m` |
| `bt2020nc` / `bt2020c` | BT.2020 | `bt2020nc` / `bt2020` / `bt2020-10` |

**出力タグは「マトリクス系統から導出した正準の 3 つ組」であり、入力タグのフィールド単位コピーではありません。**
判定は入力の `color_space`（マトリクス）だけを見て上表の系統に振り分け、その系統に対応する
(colorspace, primaries, transfer) で再タグ付けします。したがって入力の primaries / transfer が
マトリクスと食い違っていても、出力は判定された系統の正準値に正規化されます。

> 例: 入力が `color_space=smpte170m`（BT.601）かつ `color_transfer=bt709`（SD 素材でよくある不一致タグ）
> の場合、BT.601 と判定され、出力 transfer は `smpte170m` に正規化されます（`bt709` は引き継ぎません）。
> BT.2020 のトランスファに `bt2020-10` を用いるのは、マトリクス名 `bt2020nc` が `color_trc` として無効な
> ためで、マトリクスとトランスファは別値として保持する必要があるからです。

## 出力コンテナ

**最終的なコンテナは出力ファイルの拡張子で決まります。** `.mkv` は Matroska、`.mp4` / `.mov` は
MP4/MOV を生成します（**AV1-in-MP4 も可**。HEVC/AV1 ともに `.mp4` 出力を実機確認済み）。

mux は 2 段構成です。

1. NVENC の生ストリーム（HEVC `.hevc` / AV1 `.obu`）を **mkvmerge** で中間ファイルにまとめる
   （タイムコード付与）。
2. **ffmpeg** で最終 remux — 音声を結合し、色メタデータ（matrix / primaries / transfer / range）を
   付与。映像は `-c:v copy`（再エンコードなし）。出力が `.mp4` / `.mov` の場合は `-movflags +faststart`
   を付与する。

## 制限事項

- **AV1 はファイル出力のみ。** ストリーミング（`--stream`）では AV1 を指定できません（HEVC を使用）。
  AV1 + `--stream` はエラーになります。
- **AV1 は B フレーム無効。** NVENC の AV1 は B フレームに非対応のため、内部で無効化しています。
- **HDR トランスファ特性（PQ / HLG）は非対応。** 保持するのは色空間マトリクスと原色のみで、
  トランスファ特性は引き継ぎません。HDR ソースの完全な保持が必要な用途には未対応です。
- **フルレンジ（JPEG / PC range）出力は非対応。** 変換は limited（MPEG / TV range）専用です。
  入力がフルレンジの場合はエラーになります。
- **AV1 の muxing。** NVENC(AV1) の生出力（OBU）を `.obu` 一時ファイル経由で mkvmerge に渡します。
  RTX 5060 Ti (Blackwell) + ffmpeg 8 / mkvmerge v97 で正常動作を確認済み。古い mkvmerge で OBU を
  取り込めない場合は IVF 化や ffmpeg 直 mux が必要になる可能性があります。

## GUI

設定パネルの「エンコード」セクションに **Codec**（HEVC / AV1）と **Bit Depth**（Auto / 8 / 10）の
ドロップダウンを追加しています。CLI と同じ Pipeline を使うため挙動は一致します。

## 実装メモ（開発者向け）

- RGB→サーフェス変換: `jasna/media/rgb_to_p010.py` の `chw_rgb_to_surface(frame, colorspace, bit_depth)`。
  (Kr, Kb) から limited-range マトリクスを生成し、8bit は NV12（uint8）、10bit は P010（int16, 上位10bit）を返す。
- 色空間の判定/保持: `jasna/media/__init__.py` の `Colorspace` enum と `VideoMetadata.yuv_colorspace`
  （av の `Colorspace` enum は BT.2020 を表現できないため独自に保持）。
- エンコーダ: `jasna/media/video_encoder.py` がコーデック＋ビット深度から `fmt`（P010/NV12）・`profile`・
  B フレーム数・一時ファイル拡張子（`.hevc`/`.obu`）を決定。
- フルレンジ判定: `jasna/media/__init__.py` で ffprobe の `color_range` が `pc` または `jpeg` のとき
  フルレンジ（`AvColorRange.JPEG`）とみなし、パイプライン冒頭で拒否する。`tv` / 不在 / `unknown` は
  limited（MPEG）。ffprobe はフルレンジを `pc` と報告する（`jpeg` ではない）ため、`pc` を取りこぼさない
  ことが重要。
