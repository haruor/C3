# generator/views.py
from google.api_core.exceptions import ResourceExhausted
import re

import re # ★★★ この行を追加 ★★★
import os
import uuid # ★★★ 追加 ★★★
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from PIL import Image

from google import genai

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial import Delaunay


# preprocess_image_points
def preprocess_image_points(image_path, max_stars=15):
    """
    画像から点列を抽出し、整形・正規化。
    Vmagのダミー情報も追加し、max_stars * 3 のフラットなベクトルを返す。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    max_dim = 800
    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = np.column_stack(np.where(binary == 0)) # (y, x)

    if len(points) == 0:
        print("警告: 画像に黒いピクセルがありませんでした。ダミーの3次元ベクトルを返します。")
        # 0のVmagダミー列を含む、max_stars * 3 のダミーベクトルを返す
        return torch.tensor(np.zeros((max_stars, 3)).flatten(), dtype=torch.float32)

    max_points_for_kmeans = 50000
    if len(points) > max_points_for_kmeans:
        indices = np.random.choice(len(points), max_points_for_kmeans, replace=False)
        points = points[indices]

    coords_to_normalize = points.astype(np.float32) # (N, 2) float32

    # 座標を正規化（0-1スケールに）
    data_max = coords_to_normalize.max(axis=0) # (max_y, max_x)
    data_max[data_max == 0] = 1.0 # ゼロ除算を避ける
    coords_normalized = coords_to_normalize / data_max # (N, 2) normalized coords

    # ★★★ ここからVmagのダミー列を追加して、3次元にする ★★★
    # Vmagのダミー列を作成（モデルがVmagの0-1正規化範囲で学習しているので、中間値などが安全）
    dummy_vmag_column = np.full((len(coords_normalized), 1), 0.5) # 全て0.5のVmagとして扱う
    
    # 座標とダミーVmagを結合
    coords_with_dummy_vmag = np.hstack((coords_normalized, dummy_vmag_column)) # (N, 3)

    # スター数を max_stars に制限（少なければパディング）
    coords_limited = coords_with_dummy_vmag[:max_stars] # (up to max_stars, 3)
    if len(coords_limited) < max_stars:
        padding = np.zeros((max_stars - len(coords_limited), 3)) # 3次元でパディング
        coords_limited = np.vstack([coords_limited, padding])
    elif len(coords_limited) > max_stars:
        coords_limited = coords_limited[:max_stars] # Max_starsを超えたら切り捨て

    return torch.tensor(coords_limited.flatten(), dtype=torch.float32)


# ConstellationAutoEncoder クラス
class ConstellationAutoEncoder(nn.Module):
    def __init__(self, input_size, latent_dim=16):
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
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


# generate_new_constellation 関数を修正
def generate_new_constellation(image_vec, model, max_stars=15):
    """
    入力画像の点列から創作星座を生成（オートエンコーダのdecoderを使う）
    モデルの出力は (RA, Dec, Vmag) の flatten された形式を想定する。
    """
    model.eval()
    with torch.no_grad():
        latent = model.encoder(image_vec.unsqueeze(0))
        output = model.decoder(latent)
    
    # ★★★ ここを修正: (max_stars, 3) に reshape する ★★★
    coords_and_vmag = output.squeeze().reshape(max_stars, 3).numpy()
    
    # 座標とVmag情報を分離
    coords = coords_and_vmag[:, :2] # RA, Dec のみ
    vmags = coords_and_vmag[:, 2]   # Vmag のみ (これはモデルが再構成した仮想的なVmag)

    return coords, vmags # 座標とVmagを両方返すように変更


# draw_constellation_from_coords (描画関数 - サイズ引数対応)
def draw_constellation_from_coords(coords, figname, result_folder, sizes=None):
    # 1. Figureオブジェクトを作成し、サイズと比率を固定します
    fig = plt.figure(figsize=(5, 5))

    # 2. Figure全体の背景色を黒に設定します
    fig.patch.set_facecolor('black')

    # 3. 描画エリア(Axes)を追加し、その背景も黒にします
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('black')

    # --- ここから星と線を描画するロジック (変更なし) ---
    if sizes is not None and len(sizes) == len(coords):
        max_vmag = np.max(sizes) if len(sizes) > 0 else 1.0
        min_vmag = np.min(sizes) if len(sizes) > 0 else 0.0
        min_plot_size = 20
        max_plot_size = 200
        
        if max_vmag == min_vmag:
            plot_sizes = np.full_like(sizes, (min_plot_size + max_plot_size) / 2)
        else:
            normalized_vmag = (sizes - min_vmag) / (max_vmag - min_vmag)
            plot_sizes = (1 - normalized_vmag) * (max_plot_size - min_plot_size) + min_plot_size
        
        ax.scatter(coords[:, 0], coords[:, 1], s=plot_sizes, c='yellow')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=100, c='yellow')

    if len(coords) >= 2:
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                p1 = coords[i]
                p2 = coords[j]
                dist_p1_p2 = np.linalg.norm(p1 - p2)

                is_rng_edge = True
                for k in range(len(coords)):
                    if k == i or k == j:
                        continue
                    pk = coords[k]
                    dist_p1_pk = np.linalg.norm(p1 - pk)
                    dist_p2_pk = np.linalg.norm(p2 - pk)

                    if dist_p1_pk < dist_p1_p2 and dist_p2_pk < dist_p1_p2:
                        is_rng_edge = False
                        break

                if is_rng_edge:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='blue', alpha=0.5)
    # --- ここまで描画ロジック ---

    # 4. 描画範囲を固定します
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.invert_yaxis()

    # 5. 軸を非表示にします
    ax.axis('off')

    # 6. 画像を保存します (サイズが変わらないようにオプションを調整)
    # tight_layout=True を追加し、余白を適切に管理します
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(result_folder, figname + ".png"), facecolor='black')
    plt.close(fig)


# normalize_ra_dec
def normalize_ra_dec(ra_series, dec_series):
    ra_norm = (ra_series - ra_series.min()) / (ra_series.max() - ra_series.min())
    dec_norm = (dec_series - dec_series.min()) / (dec_series.max() - dec_series.min())
    return np.stack([ra_norm.values, dec_norm.values], axis=1)


# match_virtual_points_to_real_stars (Vmagも返すように修正)
def match_virtual_points_to_real_stars(virtual_coords, real_star_coords_and_vmag):
    real_coords = real_star_coords_and_vmag[:, :2] # 座標のみをKDTreeに渡す
    real_vmags = real_star_coords_and_vmag[:, 2] # Vmagを取得

    tree = KDTree(real_coords)
    dists, indices = tree.query(virtual_coords)
    
    matched_real_coords = real_coords[indices]
    matched_real_vmags = real_vmags[indices]

    return matched_real_coords, matched_real_vmags


# read_and_process_constellation_data (Vmagも読み込むように修正)
def read_and_process_constellation_data(csv_path, max_stars=15):
    df = pd.read_csv(csv_path)

    if 'Vmag' not in df.columns:
        print("警告: CSVファイルに 'Vmag' カラムが見つかりませんでした。デフォルトの明るさを使用します。")
        df['Vmag'] = 5.0 # 例としてデフォルトVmagを設定

    grouped = df.groupby("Constellation")
    feature_list = []
    labels = []

    for name, group in grouped:
        coords_and_vmag = group[["RA(deg)", "Dec(deg)", "Vmag"]].to_numpy()

        coords_and_vmag_limited = coords_and_vmag[:max_stars]
        if len(coords_and_vmag_limited) < max_stars:
            padding = np.zeros((max_stars - len(coords_and_vmag_limited), 3))
            coords_and_vmag_limited = np.vstack([coords_and_vmag_limited, padding])
        
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(coords_and_vmag_limited)

        feature_list.append(normalized_features.flatten())
        labels.append(name)

    return np.array(feature_list), labels, df

# ★★★ ここまで新しい main0724.py の内容を貼り付けます ★★★


# Gemini API クライアントの初期化 (設定からAPIキーを読み込む)
try:
    gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    print("Gemini API Client initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini API Client: {e}")
    gemini_client = None


# グローバル変数としてモデルとデータをロード (アプリケーション起動時に一度だけ)
constellation_df = None
model = None
model_path = 'constellation_autoencoder_model.pth'
constellation_csv_path = 'stars_by_constellation_akarusa.csv'


try:
    features, labels, loaded_df = read_and_process_constellation_data(constellation_csv_path)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    constellation_df = loaded_df

    input_size = features_tensor.shape[1]
    model = ConstellationAutoEncoder(input_size, latent_dim=64)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"訓練済みモデルを {model_path} からロードしました。")
    else:
        print("警告: 訓練済みモデルが見つかりません。Webアプリの起動前に 'train_and_save_model.py' を実行してください。")
        model = None
except FileNotFoundError:
    print(f"エラー: {constellation_csv_path} または {model_path} が見つかりません。必要なファイルを生成してください。")
    constellation_df = None
    model = None
except Exception as e:
    print(f"モデルのロードまたは星座データの読み込み中にエラーが発生しました: {e}")
    import traceback
    traceback.print_exc()
    constellation_df = None
    model = None


def generate_myth(image_name, constellation_type, gemini_client):
    if not gemini_client:
        return "Gemini APIクライアントが初期化されていません。神話を生成できません。"

    prompt = f"以下の情報に基づいて、新しい星座の神話を作成してください。\n"
    prompt += f"この星座は'{image_name}'という名前の画像から生成されたもので、その形状は'{constellation_type}'星座です。\n"
    prompt += f"創造性豊かな短い神話を生成してください。"

    try:
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        # --- AIが返すテキストを綺麗にする処理 ---
        raw_text = response.text
        
        # 1. 改行をスペースに置換
        text_no_newlines = raw_text.replace('\n', ' ')
        # 2. 連続する空白を一つにまとめる
        text_single_spaced = re.sub(r'\s+', ' ', text_no_newlines)
        # 3. 全体の最初と最後の空白を削除
        clean_text = text_single_spaced.strip()
        
        # 最終的に、綺麗になった神話の文章だけを返す
        return clean_text

    except ResourceExhausted: # ★★★ 429エラーをここで捕捉 ★★★
        return "リクエストが多すぎます。1分ほど待ってから、もう一度お試しください。"
    except Exception as e:
        print(f"Gemini API呼び出し中に予期せぬエラーが発生しました: {e}")
        return f"神話の生成中に予期せぬエラーが発生しました: {e}"


def index(request):
    return render(request, 'index.html')


def upload_file(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']

        # --- ファイル名を安全なものに変換 ---
        original_filename, ext = os.path.splitext(uploaded_file.name)
        safe_filename = f"{uuid.uuid4()}{ext}"
        
        # --- ここからが修正箇所 ---
        # locationを指定しない、汎用的なストレージオブジェクトを使用
        fs = FileSystemStorage() 
        
        # 'uploads'フォルダを指定してファイルを保存
        filename = fs.save(os.path.join('uploads', safe_filename), uploaded_file)
        
        filepath = fs.path(filename)
        # --- ここまでが修正箇所 ---

        try:
            Image.open(uploaded_file).verify()
            if not uploaded_file.content_type.startswith('image/'):
                raise ValueError("無効なファイル形式です。画像ファイルをアップロードしてください。")

            image_vec = preprocess_image_points(filepath, max_stars=15)

            if constellation_df is None or model is None:
                return render(request, 'error.html', {'message': 'システムがまだ準備できていません。'})

            virtual_coords, virtual_vmags = generate_new_constellation(image_vec, model, max_stars=15)
            results_dir = os.path.join(settings.MEDIA_ROOT, 'results')
            os.makedirs(results_dir, exist_ok=True)

            # 仮想星座のファイル名と保存
            virtual_constellation_filename = "virtual_stars_constellation.png"
            draw_constellation_from_coords(virtual_coords, virtual_constellation_filename, results_dir, sizes=virtual_vmags)

            # 実在星とのマッチング
            real_coords_for_kdtree = normalize_ra_dec(constellation_df['RA(deg)'], constellation_df['Dec(deg)'])
            real_star_coords_and_original_vmag = np.stack([
                real_coords_for_kdtree[:, 0],
                real_coords_for_kdtree[:, 1],
                constellation_df['Vmag'].values
            ], axis=1)
            matched_coords, matched_vmags = match_virtual_points_to_real_stars(
                virtual_coords, real_star_coords_and_original_vmag
            )

            # 実在星の星座のファイル名と保存
            real_constellation_filename = "real_stars_constellation.png"
            draw_constellation_from_coords(matched_coords, real_constellation_filename, results_dir, sizes=matched_vmags)

            # --- URL生成部分を修正 ---
            uploaded_image_url = fs.url(filename)
            # 'results'フォルダ内の画像のURLを正しく生成
            real_image_url = fs.url(os.path.join('results', real_constellation_filename))

            myth_text = generate_myth(original_filename, "実際の星にマッチした星座", gemini_client)

            return render(request, 'results.html', {
                'uploaded_image_url': uploaded_image_url,
                'real_image': real_image_url,
                'myth_text': myth_text,
                'uploaded_image_name': uploaded_file.name
            })
        except ValueError as ve:
            return render(request, 'error.html', {'message': str(ve)})
        except FileNotFoundError as fnfe:
            return render(request, 'error.html', {'message': f"必要なファイルが見つかりません: {fnfe}"})
        except Exception as e:
            print(f"ファイルアップロード処理中に予期せずエラー: {e}")
            import traceback
            traceback.print_exc()
            return render(request, 'error.html', {'message': f'画像の処理中に予期せぬエラーが発生しました。'})

    return render(request, 'index.html')