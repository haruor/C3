const WIND_CONFIG = {
    particleCount: 3000,
    baseSpeed: 0.3,
    inletHalfGap: 50,
    minHalfGap: 10,
    buildingHeight: 120,
    sceneDepth: 200,
};

let windParticles = null;
let particlePositions, particleColors;

function getHalfGap(z) {
    if (z < 0) return WIND_CONFIG.inletHalfGap;
    if (z < 80) return WIND_CONFIG.inletHalfGap - 0.5 * z;
    return WIND_CONFIG.minHalfGap;
}

function getWindVelocityContinuity(x, z) {
    const hw = getHalfGap(z);
    const vz = WIND_CONFIG.baseSpeed * WIND_CONFIG.inletHalfGap / hw;
    const vx = z < 80 ? -0.5 * (x / hw) * vz : 0;
    return { vx, vz };
}

// Dispatches to CFD data (wind_cfd.js) when available, otherwise continuity.
// instrument.js calls getWindVelocity() — no change needed there.
function getWindVelocity(x, z) {
    if (typeof getCFDWindVelocity === 'function') {
        const v = getCFDWindVelocity(x, z);
        if (v) return v;
    }
    return getWindVelocityContinuity(x, z);
}

function spawnParticle(i) {
    const hw = WIND_CONFIG.inletHalfGap;
    particlePositions[i * 3 + 0] = (Math.random() * 2 - 1) * hw * 0.9;
    particlePositions[i * 3 + 1] = Math.random() * WIND_CONFIG.buildingHeight;
    particlePositions[i * 3 + 2] = Math.random() * -30;
}

function initWindParticles() {
    const count = WIND_CONFIG.particleCount;
    particlePositions = new Float32Array(count * 3);
    particleColors = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
        spawnParticle(i);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.6,
        vertexColors: true,
        transparent: true,
        opacity: 0.85,
        sizeAttenuation: true,
    });

    windParticles = new THREE.Points(geometry, material);
    scene.add(windParticles);
}

function updateWindParticles() {
    if (!windParticles) return;

    const count = WIND_CONFIG.particleCount;
    const maxSpeed = WIND_CONFIG.baseSpeed * WIND_CONFIG.inletHalfGap / WIND_CONFIG.minHalfGap;
    const color = new THREE.Color();

    for (let i = 0; i < count; i++) {
        let x = particlePositions[i * 3 + 0];
        let y = particlePositions[i * 3 + 1];
        let z = particlePositions[i * 3 + 2];

        if (z > WIND_CONFIG.sceneDepth) {
            spawnParticle(i);
            continue;
        }

        const { vx, vz } = getWindVelocity(x, z);
        x += vx;
        z += vz;

        const hw = getHalfGap(z);
        if (Math.abs(x) > hw) x = Math.sign(x) * hw;

        particlePositions[i * 3 + 0] = x;
        particlePositions[i * 3 + 1] = y;
        particlePositions[i * 3 + 2] = z;

        const speed = Math.sqrt(vx * vx + vz * vz);
        const t = Math.min(speed / maxSpeed, 1);
        color.setHSL((1 - t) * 0.67, 1.0, 0.5);
        particleColors[i * 3 + 0] = color.r;
        particleColors[i * 3 + 1] = color.g;
        particleColors[i * 3 + 2] = color.b;
    }

    windParticles.geometry.attributes.position.needsUpdate = true;
    windParticles.geometry.attributes.color.needsUpdate = true;
}
