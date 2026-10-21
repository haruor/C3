# TunedCity2 — Wind Visualization

## プロジェクト概要

格子状に配置した46棟のビル群による風環境をブラウザ上で3D可視化するプロジェクトです。
Three.js による風パーティクルアニメーションと、OpenFOAM RANS CFD の計算結果を統合しています。

## 使用技術

- **3D可視化:** Three.js (r128)
- **ローカルサーバー:** Python / Flask
- **CFD ソルバー:** OpenFOAM 13（Multipass Ubuntu VM 上で実行）

## セットアップ

```bash
cd TunedCity2
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## シミュレーション確認方法

### CFD 風速場モード（推奨）

```bash
cd TunedCity2
WIND_MODEL=rans_cfd .venv/bin/python3 backend/app.py
```

ブラウザで **http://127.0.0.1:5051** を開く。

DevTools Console（F12）に以下が表示されれば CFD 風速場が適用されています:

```
[wind] field loaded: rans_cfd | nx=161 nz=121
```

### 連続式モデルモード（CFD なし・簡易確認用）

```bash
.venv/bin/python3 backend/app.py
```

ブラウザで **http://127.0.0.1:5051** を開く。

## 注意事項

- `WIND_MODEL=rans_cfd` を省略すると連続式モデル（簡易計算）で動作する
- CFD 風速場は `../openfoam_case_tc2/wind_field.json` から読み込む（事前に CFD 計算が必要）
- Multipass VM（openfoam）は CFD を再計算するときのみ必要。確認だけなら不要
- ポート 5051 が使用中の場合は `lsof -ti:5051 | xargs kill -9` で解放する
- TunedCity（ポート 5050）と同時起動可能

## CFD 再計算手順（ビル形状を変更した場合）

```bash
# 1. ケース再生成
curl http://127.0.0.1:5051/api/generate-cfd-case

# 2. Multipass VM で計算実行（SSH が繋がる状態で）
multipass mount /path/to/openfoam_case_tc2 openfoam:/home/ubuntu/case_tc2  # 初回のみ
multipass exec openfoam -- bash -c \
  "source /opt/openfoam13/etc/bashrc && cd /home/ubuntu/case_tc2 && \
   blockMesh && snappyHexMesh -overwrite && foamRun -solver incompressibleFluid"

# 3. wind_field.json を更新
.venv/bin/python3 -c "
from wind_solver import RansCFDWindSolver
from pathlib import Path
RansCFDWindSolver(Path('../openfoam_case_tc2')).extract_wind_field()
"
```

Multipass VM の SSH が切れた場合はターミナルで以下を実行:

```bash
sudo ps aux | grep qemu | grep -v grep | awk '{print $2}' | xargs sudo kill -9
sleep 3
sudo launchctl stop com.canonical.multipassd
sleep 2
sudo launchctl start com.canonical.multipassd
sleep 15
multipass exec openfoam -- echo "OK"   # OK と返れば接続成功
```
