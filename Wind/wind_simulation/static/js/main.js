// UIとアプリケーションロジック

const ws = new WebSocket("ws://localhost:8000/ws");

// --- UI State ---
let currentWidth = 10;
let currentDepth = 30;
const PIPE_Y_TOP = 40;
const THICKNESS = 5;

// --- 3D Viewer Initialization ---
// window.onload で確実にDOMが読み込まれてから初期化
let viewer;
window.addEventListener('DOMContentLoaded', async () => {
    // サーバーから設定（レンジ・初期値）を取得
    try {
        const res = await fetch('/api/config');
        if (res.ok) {
            const config = await res.json();
            
            // スライダーの属性を更新
            const windSlider = document.getElementById('wind-slider');
            windSlider.min = config.WIND_MS_RANGE[0];
            windSlider.max = config.WIND_MS_RANGE[1];
            windSlider.value = config.WIND_MS;
            document.getElementById('wind-ms-display').innerText = Math.round(config.WIND_MS);

            // パイプ幅はグリッド数の10倍(mm相当)で扱っていると仮定してスライダー設定
            const widthSlider = document.getElementById('width-slider');
            widthSlider.min = config.PIPE_WIDTH_RANGE[0] * 10;
            widthSlider.max = config.PIPE_WIDTH_RANGE[1] * 10;
            widthSlider.value = config.PIPE_WIDTH * 10;
            document.getElementById('width-display').innerText = config.PIPE_WIDTH * 10;
            currentWidth = config.PIPE_WIDTH;

            const depthSlider = document.getElementById('depth-slider');
            depthSlider.min = config.PIPE_DEPTH_RANGE[0] * 10;
            depthSlider.max = config.PIPE_DEPTH_RANGE[1] * 10;
            depthSlider.value = config.PIPE_DEPTH * 10;
            document.getElementById('depth-display').innerText = config.PIPE_DEPTH * 10;
            currentDepth = config.PIPE_DEPTH;

            // 和音モード初期化
            const chordToggle = document.getElementById('chord-mode-toggle');
            if (chordToggle) {
                chordToggle.checked = config.CHORD_MODE;
                chordToggle.addEventListener('change', (e) => {
                    ws.send(JSON.stringify({ type: 'update', chord_mode: e.target.checked }));
                });
            }
        }
    } catch (e) {
        console.error("Failed to load config:", e);
    }

    viewer = new Viewer3D('canvas-container');
    // 初期パイプの描画はWebSocketからのデータ受信時に行うよう変更
});

// --- Audio Engine ---
let audioCtx;
let oscillators = [];
let gainNodes = [];
let isPlaying = false;

const updateText = (id, val) => {
    const el = document.getElementById(id);
    if(el) el.innerText = val;
};
const LBM_TO_MS = 340.0 * Math.sqrt(3);

// --- Event Listeners ---
['wind', 'width', 'depth'].forEach(id => {
    document.getElementById(`${id}-slider`).addEventListener('input', (e) => {
        let val = e.target.value;
        if (id !== 'wind') {
            updateText(`${id}-display`, val);
        }

        let payload = { type: 'update' };
        if (id === 'wind') {
            updateText('wind-ms-display', val);
            payload.u_in = parseFloat(val) / LBM_TO_MS;
        }
        if (id === 'width') {
            payload.pipe_width = parseInt(val);
        }
        if (id === 'depth') {
            payload.pipe_depth = parseInt(val);
        }
        ws.send(JSON.stringify(payload));
    });
});

const audioBtn = document.getElementById('audio-btn');
audioBtn.addEventListener('click', () => {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    // ユーザー操作時に確実にAudioContextを再開する（ブラウザの自動再生ポリシー対策）
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    isPlaying = !isPlaying;
    audioBtn.innerText = isPlaying ? "STOP AUDIO" : "PLAY AUDIO";
    audioBtn.className = isPlaying ? "active" : "";

    if (!isPlaying) {
        // 全ての音を消す
        gainNodes.forEach(gain => {
            gain.gain.setTargetAtTime(0.0, audioCtx.currentTime, 0.5);
        });
    }
});

document.getElementById('perfect-c-btn').addEventListener('click', () => {
    // Set UI sliders
    document.getElementById('wind-slider').value = 4;
    document.getElementById('width-slider').value = 100;
    // 深さ282mmにすると、開口端補正を含めてぴったり261.9Hz(C4)になります
    document.getElementById('depth-slider').value = 282;
    
    // Update displays
    updateText('wind-ms-display', '4');
    updateText('width-display', '100');
    updateText('depth-display', '282');
    
    // Set chord mode checkbox
    const chordToggle = document.getElementById('chord-mode-toggle');
    if (chordToggle) chordToggle.checked = true;
    
    // Send payload
    ws.send(JSON.stringify({
        type: 'update',
        u_in: 4 / LBM_TO_MS,
        pipe_width: 100,
        pipe_depth: 282,
        chord_mode: true
    }));
});

// Base64デコード用ユーティリティ
function base64ToFloat32Array(base64) {
    const binary_string = window.atob(base64);
    const len = binary_string.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
}

// --- WebSocket Event Processing ---
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // 3Dビューアの流体とパイプ更新
    if (viewer && data.u_data) {
        const uArray = base64ToFloat32Array(data.u_data);
        viewer.updateFluid(uArray, data.nx, data.ny);
        if (data.pipes) {
            viewer.updatePipes(data.pipes, PIPE_Y_TOP, THICKNESS);
        }
    }

    updateText('step-val', `Step: ${data.step}`);
    
    // 局所風速（複数ある場合は平均値などで代表する、もしくは配列として表示）
    if (data.local_v_list && data.local_v_list.length > 0) {
        const maxV = Math.max(...data.local_v_list);
        updateText('local-v-val', maxV.toFixed(4));
        let localMs = (maxV * LBM_TO_MS).toFixed(1);
        updateText('local-v-ms-val', `(Max: ${localMs} m/s)`);
    }

    const badge = document.getElementById('resonance-badge');

    if (data.freqs && data.is_blowing_list && data.pipes) {
        // オシレーターの数がパイプの数と合わない場合は再生成
        if (audioCtx && isPlaying) {
            while (oscillators.length < data.pipes.length) {
                let osc = audioCtx.createOscillator();
                let gain = audioCtx.createGain();
                osc.type = 'sine'; // パイプオルガン風
                gain.gain.value = 0;
                osc.connect(gain);
                gain.connect(audioCtx.destination);
                osc.start();
                oscillators.push(osc);
                gainNodes.push(gain);
            }
        }

        let blowingCount = 0;
        let freqStrs = [];

        for (let i = 0; i < data.pipes.length; i++) {
            let isBlowing = data.is_blowing_list[i];
            let freq = data.freqs[i];
            let volume = data.volumes ? data.volumes[i] : 0.0;

            if (isBlowing) {
                blowingCount++;
                freqStrs.push(freq.toFixed(1));
                if (isPlaying && oscillators[i]) {
                    // 周波数がジャンプしたときに滑らかに移行
                    oscillators[i].frequency.setTargetAtTime(freq, audioCtx.currentTime, 0.1);
                    // ダイナミック音量調整（本数で割ってクリップ防止）
                    const baseGain = volume * 0.6;
                    // 全体のゲインが1.0を超えると音が割れる（クリッピング）ので制限する
                    const maxAllowedGain = 0.9 / data.pipes.length;
                    const gainVol = Math.min(baseGain, maxAllowedGain);
                    gainNodes[i].gain.setTargetAtTime(gainVol, audioCtx.currentTime, 0.1);
                }
            } else {
                if (isPlaying && oscillators[i]) {
                    gainNodes[i].gain.setTargetAtTime(0.0, audioCtx.currentTime, 0.5);
                }
            }
        }

        // 使わなくなった余分なオシレーターは音を消す
        for(let i = data.pipes.length; i < gainNodes.length; i++) {
            if (isPlaying && gainNodes[i]) {
                gainNodes[i].gain.setTargetAtTime(0.0, audioCtx.currentTime, 0.5);
            }
        }

        if (blowingCount > 0) {
            updateText('freq-val', freqStrs.join(" / "));
            badge.innerText = `パイプ共鳴中！ (${freqStrs.join(", ")} Hz)`;
            badge.className = "status-badge active";
        } else {
            updateText('freq-val', "---");
            badge.innerText = "風切り音のみ (非発音)";
            badge.className = "status-badge";
        }
    }
};
