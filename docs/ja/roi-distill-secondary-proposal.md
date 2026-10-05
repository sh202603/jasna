# 提案: 軽量な5フレームROI Distill二次復元器

[English](../en/roi-distill-secondary-proposal.md)

これは**設計提案と参考アダプター**であり、Jasnaで動く完成済み機能ではありません。
調査対象はコミット`4120fac8e749a12afa72c2f13319cad261949de1`の`modi`パイプラインです。
checkpointの重み、動画、データセット、個人パス、支援者モデルの保護処理の変更は含みません。

## 目的と組み込み位置

BasicVSR++の後へ、任意で使用できる小型PyTorch二次復元器を追加します。
復元済みのRGB ROIを5フレーム入力し、中央のROIを256pxから512pxへ補強します。
検出、追跡、一次復元器は変更しません。出力動画の解像度とFPSも既存パイプラインの設定に従います。
動画全体の4K化や、対象を分離する処理の変更は行いません。

ローカルの入出力検証に使用したモデルは、24チャンネル・12残差ブロック・5フレーム入力・
2倍出力で、パラメーター数は130,860です。演算はConv2d、ReLU、残差加算、PixelShuffleです。
中央入力のbilinear 2倍画像へ、学習した差分を足します。
拡散処理、明示的な光学フロー、前フレーム出力の再帰的な状態保持はありません。

既存の同期`SecondaryRestorer`へ、次の入出力で接続できます。

- アダプターへの一次復元入力: float RGB `[T,3,256,256]`、範囲`[0,1]`。
- モデル入力: FP32 `[B,15,256,256]`。`t−2,t−1,t,t＋1,t＋2`の順にRGBを並べます。
- モデル出力: FP32 `[B,3,512,512]`。中央フレームの画像だけです。
- アダプターの戻り値: 有効範囲へ制限した`[keep_start,keep_end)`について、
  順序どおりに並んだuint8 RGB `[3,512,512]`テンソルのリスト。

既存の`RestorationPipeline._run_secondary`、`scale_offsets`、`BlendBuffer`には、
この二次復元出力・倍率を扱う仕組みがあります。
モデルはセッション中読み込んだままにし、同じGPUプロセスで動かします。
毎フレームのcheckpoint読み出し、CPU往復、動画の中間保存は追加しません。

## 参考アダプター

以下の`model`には、**同じモデル構造**のインスタンスを渡します。
信頼できるcheckpointを`weights_only=True`で読み、
`load_state_dict(..., strict=True)`で重みを読み込み済みのものです。
この例にモデル定義、checkpointローダー、機能登録、CLI変更は含めていません。
任意の`best.pt`を選択するだけでは動きません。

最初はFP32 eager実行・バッチ1・ビュー平滑化なしで、数値の基準を確認します。
精度や座標変換を変更するのは、その確認後です。

```python
import torch

class RoiDistillSecondaryRestorer:
    name = "roi-distill"
    num_workers = 1
    prefers_cpu_input = False

    def __init__(self, model, device, view_window=0):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device, dtype=torch.float32)
        self.view_smoothing_window = int(view_window)

    @torch.inference_mode()
    def restore(self, frames_256, *, keep_start, keep_end):
        if frames_256.ndim != 4 or tuple(frames_256.shape[1:]) != (3, 256, 256):
            raise ValueError("expected RGB [T,3,256,256]")
        if not frames_256.is_floating_point():
            raise ValueError("expected float RGB in [0,1], not uint8/BGR")
        t = len(frames_256)
        ks, ke = max(0, int(keep_start)), min(t, int(keep_end))
        if ks >= ke:
            return []
        results = []
        with torch.autocast(device_type=self.device.type, enabled=False):
            for i in range(ks, ke):
                idx = [min(t - 1, max(0, i + d)) for d in (-2, -1, 0, 1, 2)]
                window = frames_256[idx].to(self.device, dtype=torch.float32)
                if not torch.isfinite(window).all():
                    raise ValueError("non-finite ROI input")
                x = window.clamp(0, 1).reshape(1, 15, 256, 256)
                y = self.model(x)
                if tuple(y.shape) != (1, 3, 512, 512) or not torch.isfinite(y).all():
                    raise ValueError("invalid ROI Distill output")
                results.append(y[0].clamp(0, 1).mul(255).round().to(torch.uint8))
        return results
```

これは正しさの確認を優先した例です。各フレームでの有限値検査はCUDAの同期を起こす可能性があり、
処理速度を最適化した実装ではありません。
後で少数の出力対象をバッチ化できますが、長いクリップの全時間窓を一度に積む方法は避けます。

## 時間境界とROI座標

同じROIクリップの前後フレームを使い、クリップの端だけ最も近い利用可能なフレームを繰り返します。
5枚のクリップでは、先頭の窓は`[0,0,0,1,2]`、末尾は`[2,3,4,4,4]`です。
1枚だけのクリップでも1枚だけ出力できます。このアダプターに短い検出の除外ルールは追加しません。

keep範囲外のフレームも入力文脈として保持します。それら自身の二次復元出力は不要ですが、
既存のクロスフェード範囲では両方のクリップの復元結果が必要になります。
別ROIや場面転換をまたいで履歴を共有しません。
モデルは5つの入力枠から中央1枚を出すため、内部で28枚分のウォームアップ出力を作る必要はありません。

Jasnaの検出継続時間フィルターや、検出途切れを補う設定は別の処理です。
補間が有効なら、クリップに合成した観測が含まれる場合があります。
現在の二次復元インターフェースには検出マスクや元のフレーム番号が渡らないので、
マスクに応じた特別なスキップ処理を行うには、メタデータの受け渡しを追加する必要があります。

`view_smoothing_window`では既存の`tracking/crop_view.py`を再利用できます。
パイプラインが平滑化した256pxビューを作り、`view_placements`を保持して、
補強後のビューから直接、映像上のbboxへサンプリングします。
アダプター内部でも同じ座標変換を戻してはいけません。まず窓0/5/15を比較する案です。

この既存の画像全体を扱うビュー合成は、補強差分だけを元の一次復元画像へ戻す合成と同一ではありません。
AppleのVideoToolboxによる時間方向フィルターも含みません。
時間的な安定性や境界の品質は、checkpointの交換だけで保証せず動画で検証する必要があります。

## 採用する場合の追加実装

1. 推論専用の小さなモデル定義・ローダーを追加します。checkpointの版、構造、重みのキー・形状、
   有限値を検査します。学習スクリプト全体を実行時にimportしたり、12ブロックの重みを
   6ブロックのモデルへ`strict=False`で読み込んだりしません。
2. 同期アダプターを追加し、`SessionConfig`、`_build_secondary_restorer`、CLIへ`roi-distill`を登録します。
   checkpointは一次復元モデル用とは別のパス引数で指定します。
3. 既存の一次復元出力の寿命とスレッド規則を維持します。依存関係を扱わずに独立CUDA streamを追加しません。
4. 時間窓、空keep範囲、1枚のクリップ、出力倍率、padding、クロスフェード、複数ROI、場面転換をテストします。
5. 後でGUI、設定保存、セッションキーによる再構築へ対応します。同じパスのモデルを交換した場合も、
   読み込み済みセッションやキャッシュを無効にする必要があります。
6. 同じ一次・二次復元入力と短い動画全体を比較した後、crop-fps、動画全体のvideo-fps、最大VRAMを別々に測ります。
7. FP32の基準を確認してから少数バッチやTensorRT/FP16を検討します。量子化や精度変更は別途画質を検証します。

このアダプターにCore ML、SwiftVR推論、DLoRAL環境への新しい依存は必要ありません。
ただし対応するPyTorchモデル定義と互換重みは必要です。
既存の暗号化`unet-4x`を置き換えたり、保護処理を回避したりするものではなく、独立した任意の方式です。

## 確認済みの内容と制限

信頼できる対応checkpointを用い、**CPUの合成入力**で以下を確認しました。

- 先頭・中央・末尾の時間窓での出力形状と有限値。
- `T=1/2/5`、部分keep範囲、空keep範囲、`T=0`での出力枚数。
- 同じ時間窓を入力した元のFP32モデルとのuint8出力の完全一致。
- 不正な画像寸法、uint8入力、NaN入力の拒否。
- 同期のサンプルクラスが実際の`SecondaryRestorer`へ適合し、
  `AsyncSecondaryRestorer`へ誤分類されないこと。
- Jasnaの実際のcrop準備→平滑化ビュー→モデル→bboxサンプリング。
  paddingと有効領域の座標を2倍へ換算する処理も含みます。

機能登録、Jasna全体でのCUDA実行、実動画の細部・ちらつき・境界、速度、VRAM、
FP16/TensorRTはまだ検証していません。
このPRはcheckpointを配布せず、失われた細部の完全復元、画質の同等性、実測した高速化を主張しません。
重みを配布する場合、モデル・教師・学習素材の利用条件を別途確認する必要があります。

現在の追跡処理は復元前に重なった枠を結合する場合があります。
大きな結合ROIを256pxへ縮小した際の情報損失を、二次エンハンサーだけで解消することはできません。
画質と性能の変化を切り分けられるよう、追跡の変更は別の作業として扱います。

ご意見をいただきたい点: 小型・任意・同一プロセスの二次復元器は`modi`で有用な方向でしょうか。
また、既存の`SecondaryRestorer`とcrop-viewの能力属性を使う位置が、適切な組み込み先でしょうか。
