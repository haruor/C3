import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

# Add repo root to path so wind_solver can be imported from 2026-C3/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from wind_solver import ContinuityWindSolver, RansCFDWindSolver

# ---------------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------------

frontend_dir = str(Path(__file__).resolve().parent.parent / "frontend")
app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
CORS(app)

# ---------------------------------------------------------------------------
# Wind model config
# ---------------------------------------------------------------------------

WIND_MODEL = os.getenv("WIND_MODEL", "continuity")   # "continuity" | "rans_cfd"
_BUILDINGS_FILE = _REPO_ROOT / "townmap" / "buildings_tc.json"
_OPENFOAM_CASE = _REPO_ROOT / "openfoam_case"

_wind_field_cache: dict | None = None


def _get_wind_field() -> dict:
    global _wind_field_cache
    if _wind_field_cache is not None:
        return _wind_field_cache

    if WIND_MODEL == "rans_cfd":
        wf_path = _OPENFOAM_CASE / "wind_field.json"
        if wf_path.exists():
            raw = json.loads(wf_path.read_text())
        else:
            print("[wind] wind_field.json not found, falling back to continuity model")
            raw = ContinuityWindSolver().export_grid()
    else:
        raw = ContinuityWindSolver().export_grid()

    # Wind speed scaling: scale ux/uz by (desired / reference) so that
    # changing inlet_velocity_ms in buildings_tc.json and restarting the
    # server changes the effective wind speed without re-running CFD.
    buildings_data = json.loads(_BUILDINGS_FILE.read_text())
    desired_ms = buildings_data.get("inlet_velocity_ms", 10.0)
    reference_ms = raw.get("inlet_velocity_ms", 10.0)
    if abs(desired_ms - reference_ms) > 1e-6:
        scale = desired_ms / reference_ms
        raw = {**raw,
               "ux": [v * scale for v in raw["ux"]],
               "uz": [v * scale for v in raw["uz"]],
               "inlet_velocity_ms": desired_ms}
        print(f"[wind] scaled {reference_ms} m/s → {desired_ms} m/s (×{scale:.3f})")

    _wind_field_cache = raw
    return _wind_field_cache


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/wind-config")
def wind_config():
    """Return which wind model is active."""
    return jsonify({"model": WIND_MODEL})


@app.route("/api/wind-field")
def wind_field():
    """Return the 2-D wind field grid for the frontend.

    Format: { model, grid: {x_min,x_max,x_step,nx,z_min,z_max,z_step,nz}, ux[], uz[] }
    ux/uz are flat lists in row-major order (index = iz*nx + ix).
    """
    return jsonify(_get_wind_field())


@app.route("/api/wind-data")
def wind_data():
    """Legacy placeholder — kept for compatibility."""
    return jsonify({"status": "ok", "speed": 10})


@app.route("/api/generate-cfd-case")
def generate_cfd_case():
    """Generate the OpenFOAM case directory from buildings_tc.json.
    Does NOT run simpleFoam — call this to prepare the case for manual execution.
    """
    if not _BUILDINGS_FILE.exists():
        return jsonify({"error": f"{_BUILDINGS_FILE} not found"}), 404

    data = json.loads(_BUILDINGS_FILE.read_text())
    solver = RansCFDWindSolver(case_dir=_OPENFOAM_CASE)
    case_dir = solver.generate_case(
        buildings=data["buildings"],
        domain=data["domain"],
        inlet_velocity_ms=data.get("inlet_velocity_ms", 10.0),
    )
    return jsonify({"status": "ok", "case_dir": str(case_dir)})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
