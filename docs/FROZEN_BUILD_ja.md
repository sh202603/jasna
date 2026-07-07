# 凍結バイナリのビルド方法（Windows、実験的）

このフォークは、スタンドアロンの**凍結配布物**（Python の導入なしで動く配布用ディレクトリ）を `dist_nuitka\jasna\` に生成できる。
実行ファイルは `jasna.exe` の 1 本だけで、引数を付けて起動すれば CLI、引数なしで起動すれば GUI として動く（GUI 起動時はコンソールウィンドウが自動で切り離される）。

upstream はパッケージングを PyInstaller から Nuitka へ切り替えたが、そのツールは非公開の submodule（`jasna/protection`）にあり、このフォークには含まれない。
`scripts\build_nuitka.py` は、このフォークが独自に持つ自己完結のビルドスクリプトである。
現時点では Windows 専用である。

> **この配布物は、ビルドした本人（または組織内）が使うためのものであり、第三者への再配布は想定していない。**
> jasna 本体は AGPL-3.0 なので、配布すれば改変込みの完全な対応ソースを提供する義務が生じる。
> それ以前に、配布物には NVIDIA のプロプライエタリなコンポーネント（TensorRT ランタイムと `nvvfx`）が含まれており、そのライセンス条件は、この配布物ごと第三者へ渡すことを素直には許していない。
> RIFE の重み（`model_weights\rife*`）は非商用ライセンスのため、既定では配布物に含めない。
> 自分で使うビルドに含めたい場合に限り `--bundle-rife` を指定する。

---

## 1. 前提条件

- [BUILDING_WINDOWS_ja.md](BUILDING_WINDOWS_ja.md) の手順でソース実行できる環境が整っていること。
  venv に全ランタイム依存に加えて `[dev]` extra（`uv pip install -e .[dev]`）が入っていれば、`nuitka>=2.4` が使える。
  Nuitka は jasna を C にコンパイルするため、Visual Studio Build Tools（MSVC）も必要である。
- CUDA Toolkit 13.x がインストールされ、`CUDA_PATH` が設定されていること。
  NPP と nvJPEG のランタイム DLL（`nppc64_13.dll` から `nvjpeg64_13.dll` までの 6 本）を `%CUDA_PATH%\bin\x64` から配布物へコピーするためである。
  これらはどの pip wheel にも含まれず、しかも凍結アプリは実行時に CUDA Toolkit を意図的に `PATH` から除外する。
  そのため、この DLL を欠いたままビルドすると、CUDA Toolkit を持たない配布先で動画デコードが失敗する配布物ができてしまう。
- `model_weights\` に必須の重み 3 ファイルがあること。
  `lada_mosaic_restoration_model_generic_v1.2.pth`、`rfdetr-v5.onnx`、`lada_mosaic_detection_model_v4_fast.pt` である。
- 任意で、`ffmpeg` と `ffprobe`（メジャーバージョン 8）と `mkvmerge` が `PATH` にあること。
  見つかれば `tools\` と `mkvtoolnix\` に同梱される。
  見つからなければ警告だけ出してビルドは続行し、凍結アプリは実行時に配布先の `PATH` へフォールバックする。
- ディスク容量。完成した配布物は約 8 GB になる（大半は torch と CUDA スタックである）。

## 2. ビルド手順

プロジェクトルートで、venv の Python から実行する。

```powershell
.\.venv\Scripts\python.exe scripts\build_nuitka.py               # フルビルド
.\.venv\Scripts\python.exe scripts\build_nuitka.py --skip-nuitka # コンパイル済み exe を再利用し、同梱手順だけやり直す
.\.venv\Scripts\python.exe scripts\build_nuitka.py --bundle-rife # RIFE 重みも同梱する（自分で使うビルドに限る）
```

初回ビルドは MSVC が約 150 個の C ファイルをコンパイルするぶん時間がかかる。
結果は clcache にキャッシュされるため、2 回目以降は変更分しかコンパイルされない。
所要時間の大半は、third-party パッケージ約 8 GB のコピーである。

スクリプトは最後にスモークテストを自動実行する（`jasna.exe --version` がプロジェクトのバージョンを返すこと、`--help` が CLI の使い方を表示することを確認する）。
生成先は `dist_nuitka\jasna\` である。

## 3. ビルドの仕組み

設計の方針は一つで、jasna 自身のコードだけをコンパイルし、third-party パッケージはすべて無変換でコピーする。

- Nuitka は standalone モードで `jasna\__main__.py` をコンパイルする。
  jasna は自分のモジュールのいくつかを `importlib` 経由で読み込み、静的解析では追えないため、`--include-package=jasna` でパッケージ全体を強制的に含める。
  一方、site-packages で見つかった third-party のトップレベルパッケージはすべて `--nofollow-import-to` でコンパイル対象から外し、配布物のルートへそのままコピーする。
- この方式が成立するのは、Nuitka standalone バイナリの `sys.path` が自身のディレクトリだけを指すからである。
  ルートに平置きしたパッケージは通常どおり import できる。
  この平置きレイアウトは `jasna\packaging\windows_dll_paths.py` が前提とする配置でもある（起動時に `torch\lib` や `tensorrt_libs`、ルート直下の `*.libs` ディレクトリなどを DLL 探索パスに登録する）。
- exe はコンソールサブシステムでビルドする（`--windows-console-mode=force`）。
  CLI がシェルをブロックして stdout と stderr が機能するためであり、GUI 起動時は jasna 側が `FreeConsole()` でコンソールを切り離す（`os_utils.drop_console_window`）。
- `--deployment` を指定する。
  jasna は自分自身（`sys.executable`）をエンジンコンパイルのサブプロセスや multiprocessing の spawn として再起動するため、Nuitka が非 deployment ビルドに入れる自己実行ガードと両立しないからである。

スクリプトは、凍結ビルド固有の三つの罠に対処している。
スクリプトへ手を入れるときは、この三つを崩さないよう注意する。

1. **stdlib の自動同梱除外リスト**：Nuitka が自動で同梱する stdlib は、コンパイルされたコードが import するものに限られる。
   平置きコピーしたパッケージが実行時に行う import（たとえば torch による `unittest.mock` と `uuid` の import）は Nuitka から見えず、しかも `unittest` や `logging` などの広範な除外リストは自動同梱されない。
   そこでスクリプトは、除外リストのうちこのプラットフォームに存在するものを、明白に不要なもの（`STDLIB_INCLUDE_SKIP`）を除いてすべて明示的に含めている。
2. **`python3.dll`**：Nuitka は `python313.dll` を同梱するが、stable ABI 用のフォワーダである `python3.dll` は同梱しない。
   limited API でビルドされた拡張（たとえば psutil の `_psutil_windows.pyd`）は `python3.dll` にリンクしており、これがないとロードに失敗する。
   スクリプトはベースの CPython インストールからコピーする。
3. **NPP と nvJPEG の DLL**：前提条件の節で述べたとおりである。

TensorRT エンジン（`*.engine` と `*_sub_engines\`）は意図的に同梱しない。
エンジンは GPU と TensorRT のバージョンに固有であり、配布先の初回起動時に `model_weights\` へ再生成されるからである（初回のみ 15 分から 60 分程度のコンパイルが走る）。

## 4. 配布物のレイアウト

```
dist_nuitka\jasna\
├── jasna.exe               # 唯一のエントリポイント（引数あり → CLI、なし → GUI）
├── python313.dll, python3.dll, vcruntime140*.dll
├── npp*_13.dll, nvjpeg64_13.dll   # CUDA Toolkit 由来のランタイム DLL
├── torch\, torchvision\, tensorrt_libs\, python_vali\, PyNvVideoCodec\, nvvfx\, ...
├── numpy.libs\, scipy.libs\, av.libs\    # ルート直下に置く必要がある（DLL 探索の前提）
├── *.dist-info\            # importlib.metadata のバージョン照会用に残す
├── tcl\, tk\               # tkinter のデータ（tk-inter プラグイン）
├── model_weights\          # 重みのみ。エンジンは含まない
├── assets\                 # テストクリップ
├── tools\                  # 同梱 ffmpeg と ffprobe（ビルド時に見つかった場合）
└── mkvtoolnix\             # 同梱 mkvmerge（ビルド時に見つかった場合）
```

## 5. 凍結ビルドの制約

- `unet-4x` と SD1.5 inpaint とライセンス認証は動かない。
  これらは非公開 submodule の `jasna/protection` を必要とし、このフォークでは空だからである。
  ソース実行時と同じ制約であり、既定のパイプライン（検出、BasicVSR++、セカンダリの rtx-super-res と tvai）には影響しない。
- FP8 recon（`--fp8-recon`）は凍結ビルドでは未検証である。
  配布先マシンで triton がカーネルを JIT コンパイルできることに依存するためである。
- 配布物は、書き込み可能で ASCII のみのパスに展開する必要がある。
  TRT エンジンが exe の隣の `model_weights\` に書き込まれるためであり、ASCII 制約はソース実行時と同じ RTX Super-Res の制限である。
- 配布先にも NVIDIA GPU（compute capability 7.5 以上）とドライバ 590 以上は必要である。
  `ffmpeg` と `mkvmerge` は、ビルド時に同梱されなかった場合に限り配布先の `PATH` に必要となる。

## 6. ビルドの検証

```powershell
cd dist_nuitka\jasna
.\jasna.exe --version                 # -> 0.7.2+modi（スクリプトが自動で確認済み）
.\jasna.exe --input assets\test_clip1_1080p.mp4 --output %TEMP%\out.mkv
```

初回の実処理では、エンジンコンパイルのサブプロセス経路（`jasna.exe --compile-engines ...` として自分自身を再起動する）も同時に検証される。
続いて `jasna.exe` をダブルクリックし、コンソールウィンドウが消えて GUI が表示されることを確認する。
配布先に近い条件で試すには、venv と CUDA Toolkit のどちらも `PATH` に含まれないマシンかシェルで実行するとよい。
同梱した DLL だけで動くことが確認できる。

## 7. トラブルシューティング

- **GPU があるのに `Error: No CUDA device` になる**：このメッセージは誤解を招くことがある。
  `os_utils.check_nvidia_gpu()` は `ImportError` を握りつぶすため、凍結アプリ内で torch の import が失敗した場合（典型的には stdlib モジュールの欠落）も「no CUDA」と報告される。
  GPU スタックを疑う前に、まずモジュール欠落（次項）を確認する。
- **`ModuleNotFoundError: <stdlib モジュール>`**：そのモジュールが Nuitka の自動同梱除外リストに載っていて、同梱から漏れている。
  `scripts\build_nuitka.py` の `STDLIB_INCLUDE_SKIP` から外す（または include ロジックを広げる）ことで解消する。
  再ビルドでコンパイルし直すのは jasna だけなので、短時間で済む。
- **`ImportError: DLL load failed while importing <拡張>`**：その拡張の依存 DLL を調べる（`pefile` などを使う）。
  `python3.dll` にリンクしていれば、フォワーダが配布物ルートにないのが原因である。
  そうでなければ、依存 DLL が凍結アプリの探索パス（配布物ルート、`torch\lib`、`*.libs`、System32）にない。
- **`... nppicc64_13.dll was not found`（ほかの npp 系や nvjpeg でも同様）**：ビルドマシンでスクリプト実行時に `CUDA_PATH` が未設定だった。
  CUDA Toolkit を入れた状態でビルドし直す。
- **nofollow のロジックを変えたあとの原因不明な誤動作**：`build\nuitka\report.xml` を確認する。
  site-packages 由来のモジュールがコンパイルされていた場合、スクリプトが警告を出す。
  third-party コードはコピーすべきであってコンパイルしてはならず、たとえば `cv2` はコンパイルすると壊れる。
