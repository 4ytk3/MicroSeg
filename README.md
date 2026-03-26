# MicroSeg

MicroSeg は、**電子顕微鏡画像（SEM / TEM）向けの粒子セグメンテーション・解析アプリ**です。  
Python を知らない人でも、配布版をダウンロードして起動できます。

---

## はじめて使う人向け（最短3ステップ）

1. `Releases` から自分のOS版をダウンロード
2. zip/tar.gz を展開
3. 実行ファイルを起動

- Windows: `microseg.exe`
- macOS: `microseg`（または `.app` 配布時は `.app`）
- Linux: `microseg`

起動後は、画像選択ダイアログで解析画像を選んで開始します。

---

## 何ができるか

- インスタンスマスク作成・編集
  - `SAM` プロンプト（正例/負例）
  - `LoRA` 推論
  - `Polygon` 手動マスク
  - 追加/削除/Undo/保存、複数マスク選択
- 粒子解析
  - Instances テーブル（ソート対応）
  - Statistics / Graphs（サイズ・面積・近接距離・フラクタルなど）
  - 単位切替（nm / um / mm）
- 評価
  - 予測マスク vs GT ROI（Current / All）
- フィルタ
  - Spatial
  - Frequency（FFT / IFFT, low/high/band/symmetric notch）

---

## 典型ワークフロー

1. 画像を読み込む
2. SAM/LoRA/Polygon でマスク作成
3. `Set` で確定し `Save` / `Save All`
4. `Analyze` で統計確認
5. GT があれば `Evaluate`

---

## 配布版を作る人向け（開発者）

### 前提

- Python 3.12+
- OSごとにそのOS上でビルド（クロスビルド不可）

### ワンコマンドビルド

#### Linux

```bash
bash scripts/build_linux.sh
```

#### macOS

```bash
bash scripts/build_macos.sh
```

#### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1
```

出力:

- `dist/microseg/microseg`（Windowsは `microseg.exe`）

### GitHub Actions 自動ビルド

- Workflow: `.github/workflows/build.yml`
- 実行条件:
  - タグ `v*` push
  - `workflow_dispatch`（手動実行）
- 生成物:
  - OS別 artifact

---

## ソース実行（開発）

```bash
python -m pip install -e .
microseg
```

または:

```bash
python -m microseg.main
```

代表オプション:

```bash
microseg --image data/sample.jpg
microseg --image-dir data/SEM_20260227
microseg --image-dir data/SEM_20260227 --lora-checkpoint checkpoints/sam_lora_epoch050.pt
```

主な引数:

- `--image`: 画像ファイル指定（複数可）
- `--image-dir`: 画像フォルダ指定
- `--init-mask-id` / `--init-mask-dir`: 初期マスク
- `--lora-checkpoint`: LoRAチェックポイント
- `--output-dir`: 出力先（デフォルト: `outputs/microseg`）

---

## 出力ファイル（例）

- `instance_ids.*`: インスタンスIDマスク
- `instance_masks/`: 個別マスク
- `results.json`: セッション結果
- `instance_prompts.json`: プロンプト情報
- `*.csv`: 統計エクスポート


## Releasesの命名規則（推奨）

利用者が迷わないように、配布ファイル名を統一します。

- `microseg-windows-x64-vX.Y.Z.zip`
- `microseg-macos-arm64-vX.Y.Z.zip`
- `microseg-macos-x64-vX.Y.Z.zip`
- `microseg-linux-x64-vX.Y.Z.tar.gz`

`X.Y.Z` はリリースバージョンです（例: `v0.1.0`）。

### どれを選ぶか

- Windows 11/10（一般的なPC）: `windows-x64`
- Mac（Apple Silicon: M1/M2/M3）: `macos-arm64`
- Mac（Intel CPU）: `macos-x64`
- Linux（x86_64）: `linux-x64`

### 配布時の注意

- 展開後はフォルダごと配布してください（one-folder構成）。
- OSが違う配布物は実行できません（Linux版をWindowsで実行など）。

---

## 注意点

- frozen 配布版では `Train` ワークスペースは無効（ソース実行時のみ利用可）
- SAMモデル重みは埋め込みません（ローカルキャッシュ/チェックポイント利用）
- Linux + Wayland は Qt バックエンド設定で挙動が変わる場合あり

---

## ライセンス

運用ポリシーに合わせて追記してください。
