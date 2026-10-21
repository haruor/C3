# main.py
import cv2
import numpy as np
import config
from geometry import create_pipe_mask
from lbm_core import init_simulation, lbm_step
from acoustics import calc_pipe_resonance, is_blowing_hard_enough

def display_realtime(u, mask, step, freq):
    """OpenCVを用いて流速（渦度）をリアルタイムにウィンドウ描画する"""
    
    # 渦度の計算
    dv_dx = np.roll(u[1], -1, axis=1) - np.roll(u[1], 1, axis=1)
    du_dy = np.roll(u[0], -1, axis=0) - np.roll(u[0], 1, axis=0)
    vorticity = dv_dx - du_dy
    vorticity[mask] = 0.0

    vort_mag = np.abs(vorticity)
    vort_mag[mask] = 0.0

    # 渦の強さを 0〜1 に正規化 (しきい値0.05)
    norm_vort = np.clip(vort_mag / 0.05, 0, 1)
    img_8u = (norm_vort * 255).astype(np.uint8)

    # HOTカラーマップにより、無風(0)は黒、強い渦は赤〜黄〜白へと発光するようにする
    color_img = cv2.applyColorMap(img_8u, cv2.COLORMAP_HOT)

    # 円柱部分を黒で塗りつぶす
    color_img[mask] = [0, 0, 0]

    # 情報テキストを描画
    text = f"Step: {step} | Freq: {freq:.1f}Hz"
    cv2.putText(color_img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 見やすいように画像を拡大 (2倍)
    display_img = cv2.resize(color_img, (config.NX * 2, config.NY * 2), interpolation=cv2.INTER_NEAREST)

    # ウィンドウに表示
    cv2.imshow("LBM Real-time Simulation", display_img)

def main():
    print("--- 物理パラメータ ---")
    print(f"Reynolds Number: {config.RE}")
    print(f"Relaxation Time (Tau): {config.TAU:.4f}")
    
    mask = create_pipe_mask()
    f = init_simulation()
    
    probe_x = config.PIPE_X + config.PIPE_WIDTH // 2
    probe_y = config.PIPE_Y_TOP - 1
    probe_y = max(0, probe_y)
    
    print("\nシミュレーション開始...")
    print("※シミュレーションウィンドウを選択した状態で『q』キーを押すと終了します。")
    
    for step in range(1, config.MAX_ITERS + 1):
        f, rho, u = lbm_step(f, mask)
        
        # 描画頻度（例えば10ステップごとに描画すると動画がスムーズになります）
        # config.PLOT_FREQ ではなく 10 に固定して滑らかにします
        if step % 10 == 0:
            vx = u[0, probe_y, probe_x]
            vy = u[1, probe_y, probe_x]
            local_v = np.sqrt(vx**2 + vy**2)
            
            real_length_m = config.PIPE_DEPTH * config.DX_REAL
            freq = calc_pipe_resonance(real_length_m)
            is_blowing = is_blowing_hard_enough(local_v, threshold=0.01)
            
            # 発音していないときは表示周波数を0にする（オプション）
            display_freq = freq if is_blowing else 0.0
            
            # 動画としての描画処理
            display_realtime(u, mask, step, display_freq)
            
            # 1ミリ秒待機し、キー入力を受け付ける（'q'キーで強制終了）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ユーザー操作により終了しました。")
                break

    # 全て終わったらウィンドウを閉じる
    cv2.destroyAllWindows()
    print("シミュレーション完了。")

if __name__ == "__main__":
    main()