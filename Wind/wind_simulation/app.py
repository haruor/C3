# app.py
import asyncio

import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import config
from geometry import create_pipe_mask
from lbm_core import init_simulation, lbm_step
from acoustics import calc_actual_frequency, calc_volume

# エラーの原因だった「FastAPIアプリケーションの初期化」
app = FastAPI()

# 静的ファイルの配信設定
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTMLファイルを読み込んで返すエンドポイント
@app.get("/")
async def get():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)

# フロントエンドに設定範囲を渡すためのAPI
@app.get("/api/config")
async def get_config():
    return {
        "WIND_MS_RANGE": config.WIND_MS_RANGE,
        "PIPE_WIDTH_RANGE": config.PIPE_WIDTH_RANGE,
        "PIPE_DEPTH_RANGE": config.PIPE_DEPTH_RANGE,
        "PIPE_WIDTH": config.BASE_PIPE_WIDTH,
        "PIPE_DEPTH": config.BASE_PIPE_DEPTH,
        "WIND_MS": config.U_IN * config.LBM_TO_MS,
        "CHORD_MODE": config.CHORD_MODE
    }

# LBMの画像を生成しBase64に変換する関数
def generate_vector_field_base64(u):
    """速度ベクトル場 u (形状: 2, NY, NX) を Float32 のバイナリデータとしてBase64エンコードする"""
    u_bytes = u.astype(np.float32).tobytes()
    return base64.b64encode(u_bytes).decode('ascii')

# WebSocket通信のエンドポイント（パイプオルガン版）
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    mask = create_pipe_mask()
    f = init_simulation()
    step = 0
    
    async def receive_updates():
        nonlocal mask, f
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "update":
                    if "chord_mode" in data:
                        config.CHORD_MODE = bool(data["chord_mode"])
                    if "u_in" in data: 
                        config.U_IN = float(data["u_in"])
                    if "pipe_width" in data: 
                        w = int(data["pipe_width"] / 10.0)
                        config.BASE_PIPE_WIDTH = min(config.PIPE_WIDTH_RANGE[1], max(config.PIPE_WIDTH_RANGE[0], w))
                    if "pipe_depth" in data: 
                        d = int(data["pipe_depth"] / 10.0)
                        config.BASE_PIPE_DEPTH = min(config.PIPE_DEPTH_RANGE[1], max(config.PIPE_DEPTH_RANGE[0], d))
                        
                    # パイプ配列の再計算
                    config.PIPES = []
                    if config.CHORD_MODE:
                        for i, ratio in enumerate(config.CHORD_RATIOS):
                            pw = max(5, int(round(config.BASE_PIPE_WIDTH * ratio))) 
                            pd = max(10, int(round(config.BASE_PIPE_DEPTH * ratio)))
                            px = config.BASE_PIPE_X + i * (pw + config.PIPE_SPACING)
                            config.PIPES.append({
                                'x': int(px), 
                                'width': pw, 
                                'depth': pd,
                                'exact_width': config.BASE_PIPE_WIDTH * ratio,
                                'exact_depth': config.BASE_PIPE_DEPTH * ratio
                            })
                    else:
                        config.PIPES.append({
                            'x': config.BASE_PIPE_X, 
                            'width': config.BASE_PIPE_WIDTH, 
                            'depth': config.BASE_PIPE_DEPTH,
                            'exact_width': float(config.BASE_PIPE_WIDTH),
                            'exact_depth': float(config.BASE_PIPE_DEPTH)
                        })

                    mask = create_pipe_mask()
        except WebSocketDisconnect:
            pass

    asyncio.create_task(receive_updates())

    try:
        while True:
            for _ in range(5):
                f, rho, u = lbm_step(f, mask)
                step += 1

            freqs = []
            is_blowing_list = []
            local_v_list = []
            volumes = []

            for pipe in config.PIPES:
                # Initialize mode if not present
                if 'mode' not in pipe:
                    pipe['mode'] = 1

                # パイプの開口部の中心座標を計算（左壁の厚さを考慮）
                probe_x = pipe['x'] + config.THICKNESS + pipe['width'] // 2
                # 開口部の少し上（外側）の風速を測る
                probe_y = max(0, config.PIPE_Y_TOP - 2)
                
                local_v = np.sqrt(u[0, probe_y, probe_x]**2 + u[1, probe_y, probe_x]**2)
                if np.isnan(local_v):
                    local_v = 0.0
                local_v_list.append(float(local_v))
                
                real_length_m = pipe.get('exact_depth', pipe['depth']) * config.DX_REAL
                width_m = pipe.get('exact_width', pipe['width']) * config.DX_REAL
                local_v_ms = local_v * config.LBM_TO_MS

                global_v_ms = config.U_IN * config.LBM_TO_MS
                
                freq, new_mode, efficiency = calc_actual_frequency(
                    length_m=real_length_m, 
                    width_m=width_m, 
                    global_v_ms=global_v_ms,
                    local_v_ms=local_v_ms, 
                    current_mode=pipe['mode']
                )
                
                pipe['mode'] = new_mode
                freqs.append(float(freq))
                
                # 下流のパイプが風の影に隠れて鳴らなくなるのを防ぐため、和音演奏時はglobal_v_msをベースに音量計算する
                # 閾値は少し下げてそよ風でも鳴りやすくする
                vol = calc_volume(global_v_ms, threshold=2.0, efficiency=efficiency)
                volumes.append(float(vol))
                
                # is_blowing is True if volume is greater than 0
                is_blowing_list.append(bool(vol > 0.001))

            img_base64 = generate_vector_field_base64(u)

            await websocket.send_json({
                "step": step,
                "pipes": config.PIPES,
                "freqs": freqs,
                "is_blowing_list": is_blowing_list,
                "local_v_list": local_v_list,
                "volumes": volumes,
                "u_data": img_base64,
                "nx": config.NX,
                "ny": config.NY
            })
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print("クライアントが切断しました。")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)