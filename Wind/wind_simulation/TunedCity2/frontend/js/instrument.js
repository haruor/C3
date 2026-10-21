// パイプオルガン — acoustics.py / config.py から完全移植
//
// オルガンは2基構成:
//   MAIN … 基本音 248 Hz / 第3倍音 745 Hz
//   SUB  … MAIN の1オクターブ上 496 Hz を基本音とし、MAIN の共鳴が
//           落ち込む風速 6.5〜8.1 m/s のデッドゾーンを埋める

const INSTRUMENT_CONFIG = {
    DX_REAL: 0.01,
    SPEED_OF_SOUND: 340.0,
    BASE_PIPE_WIDTH: 10,                  // config.py BASE_PIPE_WIDTH=10
    BASE_PIPE_DEPTH: 30,                  // config.py BASE_PIPE_DEPTH=30 → L=0.3m
    CHORD_RATIOS: [1.0, 0.7937, 0.6674], // config.py CHORD_RATIOS (平均律 ド:ミ:ソ)
    BLOW_THRESHOLD_MS: 2.0,              // app.py: threshold=2.0
    MAX_VZ: 1.5,
    REFERENCE_WIND_MS: 10.0,
    position: { x: 0, y: 0, z: 120 },

    SUB_ORGAN_ENABLED: true,
    // MAIN群 / SUB群 は前後(Z)に、和音の3本は左右(X)に並べる。
    // 路地は z≧80 で x ∈ [-10, +10] しかないため、和音の左右幅は
    // 間隔6 × 3本 = ±6（パイプ半幅1.5を含め ±7.5）に収めている。
    ORGAN_GROUP_SPACING_Z: 40,           // MAIN群 / SUB群 の Z 方向間隔
    PIPE_SPACING_X: 6,                   // 和音パイプ同士の X 方向間隔
    PIPE_LIT_THRESHOLD: 0.3,             // これ以上の音量で発光・「共鳴中」表示
    MASTER_GAIN: 0.6,
};

let instrumentPipes = [];  // [{ mesh, organ, widthM, lengthM, freq, volume, osc, gainNode, currentMode }]
let pipesPerOrgan = 1;     // 音量正規化と表示の行分割に使う
let audioCtx = null;
let isAudioPlaying = false;
let isChordMode = false;

// acoustics.py: calc_actual_frequency() の完全移植
function calcActualFrequency(lengthM, widthM, globalVMs, _localVMs, currentMode) {
    if (lengthM <= 0 || widthM <= 0) return { freq: 0, newMode: 1, efficiency: 0 };

    const deltaL = 0.425 * widthM;
    const effectiveLength = lengthM + deltaL;
    const f_r1 = INSTRUMENT_CONFIG.SPEED_OF_SOUND / (4.0 * effectiveLength);
    const f_r3 = 3.0 * f_r1;

    const virtualB = widthM / 32.0;
    const f_e = 0.2 * (globalVMs / virtualB);

    let newMode = currentMode;
    if (currentMode === 1 && f_e > f_r3 * 0.7) newMode = 3;
    if (currentMode === 3 && f_e < f_r1 * 1.5) newMode = 1;

    const f_n = (newMode === 1) ? f_r1 : f_r3;

    const Q = 1.5;
    const detune = (f_e - f_n) / f_n;
    let efficiency = 1.0 / (1.0 + 4.0 * Q * Q * detune * detune);
    if (efficiency < 0.2) efficiency *= 0.5;

    return { freq: f_n, newMode, efficiency };
}

// acoustics.py: calc_volume() の完全移植
function calcVolume(windVMs, threshold = 3.0, efficiency = 1.0) {
    if (windVMs <= threshold) return 0.0;
    const basePower = Math.log10(1.0 + (windVMs - threshold)) * 3.0;
    return Math.min(1.0, 0.8 * basePower * efficiency);
}

// SUB は「MAIN 基本音の2倍音 (=1オクターブ上)」を基本音とする閉管。
// 幅は必ず MAIN と同一に保つ — エッジトーン f_e = 0.2v/(W/32) を共有させ、
// 共鳴ピークを MAIN のデッドゾーン中心 7.75 m/s に一致させるため。
// 幅も相似縮小すると f_e/f_r 比が不変になり、MAIN と全く同じ風速で
// デッドゾーンが再現されてしまい効果がゼロになる。
function subPipeDepthGrid() {
    const { BASE_PIPE_WIDTH, BASE_PIPE_DEPTH, DX_REAL } = INSTRUMENT_CONFIG;
    const W = BASE_PIPE_WIDTH * DX_REAL;                    // 0.10 m
    const mainEff = BASE_PIPE_DEPTH * DX_REAL + 0.425 * W;  // 0.3425 m
    const subEff = mainEff / 2.0;                           // 0.17125 m
    return (subEff - 0.425 * W) / DX_REAL;                  // = 12.875 グリッド
}

function getOrganSpecs() {
    const specs = [{
        id: 'main', label: 'MAIN',
        depth: INSTRUMENT_CONFIG.BASE_PIPE_DEPTH,
        width: INSTRUMENT_CONFIG.BASE_PIPE_WIDTH,
        baseColor: 0x888888, litColor: 0xffaa44, litEmissive: 0x331100,
    }];
    if (INSTRUMENT_CONFIG.SUB_ORGAN_ENABLED) {
        specs.push({
            id: 'sub', label: 'SUB',
            depth: subPipeDepthGrid(),
            width: INSTRUMENT_CONFIG.BASE_PIPE_WIDTH,   // 必ず MAIN と同一
            baseColor: 0x667788, litColor: 0x44ccff, litEmissive: 0x002233,
        });
    }
    return specs;
}

function getLocalWindSpeedMs() {
    const { x, z } = INSTRUMENT_CONFIG.position;
    const { vz } = getWindVelocity(x, z);
    return (vz / INSTRUMENT_CONFIG.MAX_VZ) * INSTRUMENT_CONFIG.REFERENCE_WIND_MS;
}

function buildInstrumentPipes() {
    instrumentPipes.forEach(p => {
        scene.remove(p.mesh);
        p.mesh.geometry.dispose();
        p.mesh.material.dispose();
    });
    instrumentPipes = [];

    const ratios = isChordMode ? INSTRUMENT_CONFIG.CHORD_RATIOS : [1.0];
    const organs = getOrganSpecs();
    const { x: cx, y: cy, z: cz } = INSTRUMENT_CONFIG.position;
    pipesPerOrgan = ratios.length;

    // 群の前後順は「背の低いオルガンほど手前 (-Z = カメラ側)」とし、
    // 奥の背が高い群が隠れないようにする。SUB は MAIN より低いので常に手前。
    // instrumentPipes 配列自体は「オルガン順・根音先頭」を保つ（表示ロジックが依存）ため、
    // 位置決めにだけソート済みの並び順を使う。
    const organOrder = [...organs].sort((a, b) => a.depth - b.depth);

    organs.forEach((organ) => {
        const gi = organOrder.indexOf(organ);
        const groupZ = (gi - (organs.length - 1) / 2) * INSTRUMENT_CONFIG.ORGAN_GROUP_SPACING_Z;

        ratios.forEach((ratio, i) => {
            const widthM = organ.width * ratio * INSTRUMENT_CONFIG.DX_REAL;
            const lengthM = organ.depth * ratio * INSTRUMENT_CONFIG.DX_REAL;
            const pipeHeight = organ.depth * ratio * 2;

            const geometry = new THREE.BoxGeometry(3, pipeHeight, 3);
            const material = new THREE.MeshStandardMaterial({
                color: organ.baseColor,
                metalness: 0.5,
                roughness: 0.4,
                emissive: new THREE.Color(0x000000),
            });
            const mesh = new THREE.Mesh(geometry, material);

            // 和音の3本は左右(X)、MAIN/SUB の群は前後(Z)
            const xOffset = (i - (ratios.length - 1) / 2) * INSTRUMENT_CONFIG.PIPE_SPACING_X;
            mesh.position.set(cx + xOffset, cy + pipeHeight / 2, cz + groupZ);
            mesh.castShadow = true;
            scene.add(mesh);

            instrumentPipes.push({
                mesh, organ, widthM, lengthM,
                freq: 0, volume: 0, osc: null, gainNode: null, currentMode: 1,
            });
        });
    });
}

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();

    instrumentPipes.forEach(pipe => {
        if (!pipe.osc) {
            const osc = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.value = pipe.freq || 440;
            gainNode.gain.value = 0;
            osc.connect(gainNode);
            gainNode.connect(audioCtx.destination);
            osc.start();
            pipe.osc = osc;
            pipe.gainNode = gainNode;
        }
    });
}

function rebuildPipes() {
    instrumentPipes.forEach(p => {
        if (p.osc) { p.osc.stop(); p.osc = null; p.gainNode = null; }
    });
    buildInstrumentPipes();
    if (isAudioPlaying) initAudio();
}

function initInstrument() {
    buildInstrumentPipes();

    document.getElementById('inst-audio-btn').addEventListener('click', () => {
        isAudioPlaying = !isAudioPlaying;
        const btn = document.getElementById('inst-audio-btn');
        btn.textContent = isAudioPlaying ? 'STOP AUDIO' : 'PLAY AUDIO';
        btn.classList.toggle('inst-active', isAudioPlaying);
        if (isAudioPlaying) {
            initAudio();
        } else {
            instrumentPipes.forEach(p => {
                if (p.gainNode) p.gainNode.gain.setTargetAtTime(0, audioCtx.currentTime, 0.5);
            });
        }
    });

    document.getElementById('inst-chord-toggle').addEventListener('change', (e) => {
        isChordMode = e.target.checked;
        rebuildPipes();
    });

    const subToggle = document.getElementById('inst-sub-toggle');
    if (subToggle) {
        subToggle.checked = INSTRUMENT_CONFIG.SUB_ORGAN_ENABLED;
        subToggle.addEventListener('change', (e) => {
            INSTRUMENT_CONFIG.SUB_ORGAN_ENABLED = e.target.checked;
            rebuildPipes();
        });
    }
}

function renderInstrumentReadout(windSpeedMs) {
    const speedEl = document.getElementById('inst-wind-speed');
    if (speedEl) speedEl.textContent = windSpeedMs.toFixed(1);

    // オルガンごとに代表パイプ（各群の先頭 = 根音）を1行表示
    const box = document.getElementById('inst-organ-readout');
    if (box) {
        const lines = [];
        for (let g = 0; g < instrumentPipes.length; g += pipesPerOrgan) {
            const p = instrumentPipes[g];
            const pct = Math.round(p.volume * 100);
            const bar = '█'.repeat(Math.round(p.volume * 10)).padEnd(10, '░');
            lines.push(
                `${p.organ.label.padEnd(5)}${p.freq.toFixed(0).padStart(5)} Hz ` +
                `${bar} ${String(pct).padStart(3)}%`
            );
        }
        box.textContent = lines.join('\n');
    }

    const litPipes = instrumentPipes.filter(p => p.volume > INSTRUMENT_CONFIG.PIPE_LIT_THRESHOLD);
    const anyLit = litPipes.length > 0;

    const statusEl = document.getElementById('inst-status');
    if (statusEl) {
        statusEl.textContent = anyLit ? 'パイプ共鳴中！' : '風切り音のみ';
        statusEl.className = 'inst-status' + (anyLit ? ' inst-resonating' : '');
    }

    const freqEl = document.getElementById('inst-freq');
    if (freqEl) {
        freqEl.textContent = anyLit
            ? litPipes.map(p => p.freq.toFixed(0)).join(' / ')
            : '---';
    }
}

function updateInstrument() {
    if (instrumentPipes.length === 0) return;

    const windSpeedMs = getLocalWindSpeedMs();

    instrumentPipes.forEach(pipe => {
        const { freq, newMode, efficiency } = calcActualFrequency(
            pipe.lengthM, pipe.widthM, windSpeedMs, windSpeedMs, pipe.currentMode
        );
        pipe.currentMode = newMode;
        pipe.freq = freq;
        pipe.volume = calcVolume(windSpeedMs, INSTRUMENT_CONFIG.BLOW_THRESHOLD_MS, efficiency);

        const lit = pipe.volume > INSTRUMENT_CONFIG.PIPE_LIT_THRESHOLD;
        pipe.mesh.material.color.set(lit ? pipe.organ.litColor : pipe.organ.baseColor);
        pipe.mesh.material.emissive.set(lit ? pipe.organ.litEmissive : 0x000000);

        if (audioCtx && isAudioPlaying && pipe.osc) {
            const now = audioCtx.currentTime;
            pipe.osc.frequency.setTargetAtTime(freq, now, 0.1);
            pipe.gainNode.gain.setTargetAtTime(
                INSTRUMENT_CONFIG.MASTER_GAIN * pipe.volume / pipesPerOrgan, now, 0.1
            );
        }
    });

    renderInstrumentReadout(windSpeedMs);
}
