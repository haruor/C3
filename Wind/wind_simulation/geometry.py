# geometry.py
import numpy as np
import config

def create_pipe_mask():
    """
    グリッド上のパイプ（コの字型、上部が開口）の位置を True とした2次元ブール配列を生成する。
    複数のパイプ (config.PIPES) に対応し、すべて論理和で結合する。
    """
    # yとxの座標グリッドを作成 (yが0〜NY, xが0〜NX)
    y, x = np.mgrid[0:config.NY, 0:config.NX]
    
    mask = np.zeros((config.NY, config.NX), dtype=bool)
    
    for pipe in config.PIPES:
        px = pipe['x']
        pw = pipe['width']
        pd = pipe['depth']
        
        # 開口部が上（y=PIPE_Y_TOP）を向いているため、底面は下方向に配置される
        pipe_y_bottom = config.PIPE_Y_TOP + pd
        
        # 底面
        bottom = (y >= pipe_y_bottom) & (y < pipe_y_bottom + config.THICKNESS) & \
                 (x >= px) & (x < px + pw + 2 * config.THICKNESS)
                 
        # 左壁
        left = (y >= config.PIPE_Y_TOP) & (y < pipe_y_bottom) & \
               (x >= px) & (x < px + config.THICKNESS)
               
        # 右壁
        right = (y >= config.PIPE_Y_TOP) & (y < pipe_y_bottom) & \
                (x >= px + pw + config.THICKNESS) & (x < px + pw + 2 * config.THICKNESS)
        
        # 結合
        mask = mask | bottom | left | right
        
    return mask