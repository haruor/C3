// wind_cfd.js — /api/wind-field から風速グリッドを fetch し bilinear 補間で速度を返す
// getWindVelocity(x, z) のバックエンドとして wind.js から呼ばれる

let cfdWindField = null;
// Schema: { model, grid:{x_min,x_max,x_step,nx,z_min,z_max,z_step,nz}, ux:[], uz:[] }
// ux/uz index: iz * nx + ix  (row=z, col=x)

async function loadCFDWindField() {
    try {
        const wf = await fetch('/api/wind-field').then(r => r.json());
        cfdWindField = wf;
        console.log('[wind] field loaded:', wf.model,
                    '| nx=' + wf.grid.nx, 'nz=' + wf.grid.nz);
    } catch (e) {
        console.warn('[wind] /api/wind-field fetch failed — using continuity fallback:', e);
        cfdWindField = null;
    }
}

function getCFDWindVelocity(x, z) {
    // Returns { vx, vz } or null when data is not available (caller falls back to continuity)
    if (!cfdWindField) return null;

    const g = cfdWindField.grid;
    const nx = g.nx;

    // Clamp to grid bounds
    const xc = Math.max(g.x_min, Math.min(g.x_max, x));
    const zc = Math.max(g.z_min, Math.min(g.z_max, z));

    // Fractional grid indices
    const fx = (xc - g.x_min) / g.x_step;
    const fz = (zc - g.z_min) / g.z_step;

    const ix0 = Math.max(0, Math.min(g.nx - 2, Math.floor(fx)));
    const iz0 = Math.max(0, Math.min(g.nz - 2, Math.floor(fz)));
    const ix1 = ix0 + 1;
    const iz1 = iz0 + 1;
    const tx = fx - ix0;
    const tz = fz - iz0;

    function idx(iz, ix) { return iz * nx + ix; }

    const ux = cfdWindField.ux;
    const uz = cfdWindField.uz;

    const vx = (1 - tz) * ((1 - tx) * ux[idx(iz0, ix0)] + tx * ux[idx(iz0, ix1)])
             +      tz  * ((1 - tx) * ux[idx(iz1, ix0)] + tx * ux[idx(iz1, ix1)]);
    const vz = (1 - tz) * ((1 - tx) * uz[idx(iz0, ix0)] + tx * uz[idx(iz0, ix1)])
             +      tz  * ((1 - tx) * uz[idx(iz1, ix0)] + tx * uz[idx(iz1, ix1)]);

    // Convert m/s back to Three.js simulation units
    // API returns m/s; wind.js expects Three.js units/frame
    // INSTRUMENT_CONFIG.MAX_VZ (1.5 units/frame) = REFERENCE_WIND_MS (10 m/s)
    const MS_TO_UNITS = 1.5 / 10.0;
    return { vx: vx * MS_TO_UNITS, vz: vz * MS_TO_UNITS };
}
