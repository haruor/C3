# acoustics.py
import numpy as np
from numba import njit
import math

@njit
def calc_actual_frequency(length_m, width_m, global_v_ms, local_v_ms, current_mode, speed_of_sound=340.0):
    """
    風速、パイプの長さ、幅からエッジトーン周波数を計算し、
    ヒステリシスを考慮して鳴るべき倍音モードとその周波数、および共鳴効率を返す
    """
    if length_m <= 0 or width_m <= 0:
        return 0.0, 1, 0.0

    # 開口端補正 (End correction)
    # 穴の周囲が平坦な面（フランジ付き）の場合、補正量 ΔL は約 0.85 * 半径
    # 今回は半径 r = width_m / 2 なので、ΔL = 0.85 * (width_m / 2) = 0.425 * width_m
    delta_L = 0.425 * width_m
    effective_length = length_m + delta_L

    # パイプの共鳴周波数（閉管）
    f_r1 = speed_of_sound / (4.0 * effective_length)      # 基本音 (モード1)
    f_r3 = 3.0 * speed_of_sound / (4.0 * effective_length)  # 第3倍音 (モード3)

    # カルマン渦（エッジトーン）の周波数 (Strouhal number ~ 0.2)
    b = width_m  # 風が飛び越える穴の直径（幅）
    if b <= 0:
        b = 0.001
        
    # 物理スケールとUIのスケールを合わせるためのデフォルメ補正係数
    # 現実の管楽器の吹き口は数mmのため、UI上の巨大な幅（100mm等）をそのまま使うと非現実的な風速が必要になる。
    # そのため、エッジトーン計算時のみ仮想的に幅を約1/32に縮小して計算する
    width_scale = 1.0 / 32.0
    virtual_b = b * width_scale

    f_e = 0.2 * (global_v_ms / virtual_b)

    # ヒステリシス（デッドゾーン）によるモード遷移の判定
    new_mode = current_mode
    if current_mode == 1:
        # 基本音から第3倍音へ上がるには、渦周波数が第3倍音にかなり近づく必要がある
        if f_e > f_r3 * 0.7:
            new_mode = 3
    else: # current_mode == 3
        # 第3倍音から基本音へ下がるには、渦周波数が基本音にかなり下がる必要がある
        if f_e < f_r1 * 1.5:
            new_mode = 1

    # 現在の共鳴周波数
    f_n = f_r1 if new_mode == 1 else f_r3

    # 共鳴効率（Resonance Efficiency）の計算
    # 渦周波数 f_e と 共鳴周波数 f_n が近いほど効率が高い (ローレンツ曲線モデル)
    Q = 1.5 # 共鳴の鋭さ (Q値が大きいほどストライクゾーンが狭くなる)
    detune = (f_e - f_n) / f_n
    efficiency = 1.0 / (1.0 + 4.0 * (Q**2) * (detune**2))

    # モード切り替えの谷間（デッドゾーン）をより明確にするため、
    # f_e が f_r1 と f_r3 の中間にあるときの効率をさらに落とす
    if efficiency < 0.2:
        efficiency = efficiency * 0.5

    return f_n, new_mode, efficiency

@njit
def calc_volume(wind_v_ms, threshold=3.0, efficiency=1.0):
    """
    風速の運動エネルギーと共鳴効率から最終的な音量を計算する
    """
    if wind_v_ms <= threshold:
        return 0.0
    
    # 運動エネルギー由来の基本パワー
    # 人間の聴覚に合わせて対数カーブにし、そよ風(4m/s)でもしっかり聞こえつつ、強風でも爆音になりすぎないようにする
    base_power = math.log10(1.0 + (wind_v_ms - threshold)) * 3.0
    
    # スケール調整
    k = 0.8
    volume = k * base_power * efficiency
    
    # 音量が大きくなりすぎないようにクリップ（例: 最大1.0）
    if volume > 1.0:
        volume = 1.0
        
    return volume