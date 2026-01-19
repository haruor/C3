import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
# ★★★ この行を追加 ★★★
import numpy as np
# ★★★ ここまで追加 ★★★

# ★★★ ここに read_and_process_constellation_data 関数、ConstellationAutoEncoder クラスを貼り付けます ★★★
# (views.py に貼り付けたものと同じ定義)

# ConstellationAutoEncoder クラスもここに含まれる必要があります
class ConstellationAutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim=64): # latent_dim=16 はデフォルト、後で上書きされます
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_size),
            nn.Sigmoid() # Sigmoidも忘れずに
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# read_and_process_constellation_data 関数もここに含まれる必要があります
def read_and_process_constellation_data(csv_path, max_stars=15):
    df = pd.read_csv(csv_path)

    # Vmagが存在しない場合はデフォルト値を使用 (これも重要な部分)
    if 'Vmag' not in df.columns:
        print("警告: CSVファイルに 'Vmag' カラムが見つかりませんでした。デフォルトの明るさを使用します。")
        df['Vmag'] = 5.0 # 例としてデフォルトVmagを設定

    grouped = df.groupby("Constellation")
    feature_list = []
    labels = []

    for name, group in grouped:
        # ★★★ ここを修正し、Vmagも含めて抽出 ★★★
        coords_and_vmag = group[["RA(deg)", "Dec(deg)", "Vmag"]].to_numpy()

        # スター数を max_stars に制限（少なければパディング）
        coords_and_vmag_limited = coords_and_vmag[:max_stars]
        if len(coords_and_vmag_limited) < max_stars:
            padding = np.zeros((max_stars - len(coords_and_vmag_limited), 3)) # Vmagも考慮して3次元でパディング
            coords_and_vmag_limited = np.vstack([coords_and_vmag_limited, padding])
        
        # MinMaxScaler は特徴量ごとに独立して正規化するため、Vmag も 0-1 に変換されます。
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(coords_and_vmag_limited)

        feature_list.append(normalized_features.flatten())  # 1次元ベクトルに
        labels.append(name)

    return np.array(feature_list), labels, df


if __name__ == "__main__":
    model_path = 'constellation_autoencoder_model.pth'
    constellation_csv_path = 'stars_by_constellation_akarusa.csv' # 新しいCSVファイル名

    if not os.path.exists(constellation_csv_path):
        print(f"エラー: {constellation_csv_path} が見つかりません。最初にstars_by_constellation.pyを実行してください。")
        exit()

    # ★★★ ここを新しい関数に置き換え ★★★
    features, labels, _ = read_and_process_constellation_data(constellation_csv_path)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    input_size = features_tensor.shape[1]
    model = ConstellationAutoEncoder(input_size, latent_dim=64) # latent_dim=64 を明示的に指定
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    dataloader = DataLoader(TensorDataset(features_tensor), batch_size=8, shuffle=True)

    print("オートエンコーダの訓練を開始します...")
    for epoch in range(1000): # エポック数は多めに設定 (例: 1000)
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"訓練済みモデルを {model_path} に保存しました。")