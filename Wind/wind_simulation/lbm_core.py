# lbm_core.py
import numpy as np
from numba import njit
import config

# D2Q9モデル (2次元9方向) の離散速度ベクトル
C = np.array([
    [ 0,  0], [ 1,  0], [ 0,  1], [-1,  0], [ 0, -1],
    [ 1,  1], [-1,  1], [-1, -1], [ 1, -1]
], dtype=np.int32)

# 各方向の重み係数
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)

# バウンスバック（壁での反射）用の反転方向インデックス
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

@njit
def calc_macroscopic(f):
    """分布関数 f からマクロな密度 rho と速度 u を計算"""
    rho = np.sum(f, axis=0)
    u = np.zeros((2, f.shape[1], f.shape[2]), dtype=np.float64)
    for i in range(9):
        u[0] += C[i, 0] * f[i]
        u[1] += C[i, 1] * f[i]
    u[0] /= rho
    u[1] /= rho
    return rho, u

@njit
def calc_equilibrium(rho, u):
    """平衡分布関数 f_eq を計算"""
    f_eq = np.zeros((9, rho.shape[0], rho.shape[1]), dtype=np.float64)
    u_sq = u[0]**2 + u[1]**2
    for i in range(9):
        cu = C[i, 0] * u[0] + C[i, 1] * u[1]
        f_eq[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)
    return f_eq

def init_simulation():
    """シミュレーションの初期状態（初期の風速）を設定"""
    rho = np.ones((config.NY, config.NX), dtype=np.float64)
    u = np.zeros((2, config.NY, config.NX), dtype=np.float64)
    u[0, :, :] = config.U_IN  # 全体に初期流速を与える
    f = calc_equilibrium(rho, u)
    return f

def lbm_step(f, obstacle_mask):
    """LBMの1タイムステップを進める"""
    # 1. マクロ変数の計算
    rho, u = calc_macroscopic(f)
    
    # 2. 境界条件の適用 (流入境界: 左端)
    u[0, :, 0] = config.U_IN
    u[1, :, 0] = 0.0
    
    # 3. 衝突過程 (BGK近似)
    f_eq = calc_equilibrium(rho, u)
    f_post = f - (f - f_eq) / config.TAU
    
    # 簡易ディリクレ境界（左端を流入速度の平衡分布で固定）
    f_post[:, :, 0] = f_eq[:, :, 0]
    # 流出境界（右端での反射を防ぐため、一つ手前の列をコピー）
    f_post[:, :, -1] = f_post[:, :, -2]
    
    # 4. 障害物でのバウンスバック（反射）
    for i in range(9):
        f_post[i, obstacle_mask] = f[OPPOSITE[i], obstacle_mask]
        
    # 5. 伝播過程 (ストリーミング: 隣のセルへの移動)
    f_next = np.zeros_like(f)
    for i in range(9):
        # np.roll を使用して配列全体をシフト (y方向: axis=0, x方向: axis=1)
        shifted = np.roll(f_post[i], C[i, 0], axis=1)
        shifted = np.roll(shifted, C[i, 1], axis=0)
        f_next[i] = shifted
        
    return f_next, rho, u