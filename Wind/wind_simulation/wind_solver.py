"""
Wind solver interface for TunedCity / TunedCity2.

Coordinate mapping (Three.js ↔ OpenFOAM):
  Three.js X (spanwise)   → OpenFOAM X
  Three.js Z (streamwise) → OpenFOAM Y  (wind blows in +Y direction)
  Three.js Y (height)     → OpenFOAM Z  (vertical)

SCENE_SCALE: 1 Three.js unit = 1 metre
"""

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional

SCENE_SCALE: float = 1.0
REPO_ROOT = Path(__file__).resolve().parent
OPENFOAM_CASE_DIR = REPO_ROOT / "openfoam_case"
TOWNMAP_DIR = REPO_ROOT / "townmap"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class WindSolverBase:
    def get_wind_velocity(self, x: float, z: float) -> tuple[float, float]:
        """Return (vx, vz) in m/s at pedestrian height (1.5 m)."""
        raise NotImplementedError

    def export_grid(
        self,
        x_min: float = -150.0,
        x_max: float = 150.0,
        z_min: float = -30.0,
        z_max: float = 200.0,
        step: float = 5.0,
    ) -> dict:
        """Export wind field as a 2-D regular grid for the JavaScript frontend.

        Returns a dict with keys: model, grid, ux (flat list), uz (flat list).
        Index mapping: index = iz * nx + ix  (row = z, col = x).
        """
        xs = [x_min + i * step for i in range(int(round((x_max - x_min) / step)) + 1)]
        zs = [z_min + i * step for i in range(int(round((z_max - z_min) / step)) + 1)]
        nx, nz = len(xs), len(zs)
        ux_flat, uz_flat = [], []
        for z in zs:
            for x in xs:
                vx, vz = self.get_wind_velocity(x, z)
                ux_flat.append(round(vx, 4))
                uz_flat.append(round(vz, 4))
        return {
            "model": self.__class__.__name__,
            "grid": {
                "x_min": x_min, "x_max": x_max, "x_step": step, "nx": nx,
                "z_min": z_min, "z_max": z_max, "z_step": step, "nz": nz,
            },
            "ux": ux_flat,
            "uz": uz_flat,
        }


# ---------------------------------------------------------------------------
# Continuity-equation model (Python port of wind.js getWindVelocity)
# ---------------------------------------------------------------------------

class ContinuityWindSolver(WindSolverBase):
    """
    V1*A1 = V2*A2 continuity model — mirrors wind.js exactly.
    Kept as comparison baseline and OpenFOAM-absent fallback.
    """
    _BASE_SPEED: float = 0.3    # Three.js units / frame
    _INLET_HW: float = 50.0     # inlet half-gap (Three.js units)
    _MIN_HW: float = 10.0       # canyon half-gap
    _MAX_VZ: float = 1.5        # max vz (= BASE_SPEED * INLET_HW / MIN_HW)
    _REF_MS: float = 10.0       # m/s corresponding to MAX_VZ

    def _half_gap(self, z: float) -> float:
        if z < 0:
            return self._INLET_HW
        if z < 80:
            return self._INLET_HW - 0.5 * z
        return self._MIN_HW

    def get_wind_velocity(self, x: float, z: float) -> tuple[float, float]:
        hw = self._half_gap(z)
        vz_u = self._BASE_SPEED * self._INLET_HW / hw
        vx_u = (-0.5 * (x / hw) * vz_u) if z < 80 else 0.0
        scale = self._REF_MS / self._MAX_VZ
        return vx_u * scale, vz_u * scale


# ---------------------------------------------------------------------------
# RANS CFD solver (OpenFOAM simpleFoam / realizableKE)
# ---------------------------------------------------------------------------

class RansCFDWindSolver(WindSolverBase):
    """
    Generates an OpenFOAM simpleFoam case, optionally runs it, and serves
    the pedestrian-height (z = 1.5 m) wind field.

    generate_case()  — writes system/, constant/, 0/ (no OpenFOAM needed)
    run_solver()     — calls blockMesh + snappyHexMesh + simpleFoam
    extract_wind_field() — parses postProcessing → wind_field.json
    get_wind_velocity()  — bilinear interpolation; falls back to continuity
    """

    def __init__(self, case_dir: Optional[Path] = None):
        self.case_dir = Path(case_dir) if case_dir else OPENFOAM_CASE_DIR
        self._wind_field: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_case(
        self,
        buildings: list[dict],
        domain: dict,
        inlet_velocity_ms: float = 10.0,
    ) -> Path:
        """Write the complete OpenFOAM case directory and return its path."""
        d = self.case_dir
        for sub in [
            d / "0",
            d / "constant" / "triSurface",
            d / "system",
        ]:
            sub.mkdir(parents=True, exist_ok=True)

        # Coordinate aliases (Three.js → OpenFOAM)
        xmin = domain["x_min"]
        xmax = domain["x_max"]
        ymin = domain["z_min"]   # OF Y = ThreeJS Z (streamwise)
        ymax = domain["z_max"]
        zmax = domain["height"]  # OF Z = ThreeJS Y (vertical)
        U = inlet_velocity_ms

        # Write combined building STL
        stl_text = _buildings_to_stl(buildings)
        (d / "constant" / "triSurface" / "buildings.stl").write_text(stl_text)

        # system/
        (d / "system" / "blockMeshDict").write_text(
            _blockMeshDict(xmin, xmax, ymin, ymax, 0.0, zmax)
        )
        (d / "system" / "snappyHexMeshDict").write_text(
            _snappyHexMeshDict(buildings, location_in_mesh=(0, ymin + 10, 5))
        )
        (d / "system" / "controlDict").write_text(_controlDict())
        (d / "system" / "fvSchemes").write_text(_fvSchemes())
        (d / "system" / "fvSolution").write_text(_fvSolution())

        # 0/
        k_val, eps_val = _turbulence_init(U, I=0.05, L=10.0)
        (d / "0" / "U").write_text(_0_U(U))
        (d / "0" / "p").write_text(_0_p())
        (d / "0" / "k").write_text(_0_k(k_val))
        (d / "0" / "epsilon").write_text(_0_epsilon(eps_val))
        (d / "0" / "nut").write_text(_0_nut())

        # constant/
        (d / "constant" / "transportProperties").write_text(_transportProperties())
        (d / "constant" / "turbulenceProperties").write_text(_turbulenceProperties())

        print(f"[CFD] Case generated at: {d}")
        return d

    def run_solver(self) -> bool:
        """Run blockMesh → snappyHexMesh → simpleFoam. Returns False if unavailable."""
        if not shutil.which("blockMesh"):
            print("[CFD] OpenFOAM not found — case generation only mode.")
            return False
        for cmd in [
            ["blockMesh"],
            ["snappyHexMesh", "-overwrite"],
            ["simpleFoam"],
        ]:
            r = subprocess.run(cmd, cwd=self.case_dir, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[CFD] {' '.join(cmd)} failed:\n{r.stderr[-2000:]}")
                return False
            print(f"[CFD] {' '.join(cmd)} OK")
        return True

    def extract_wind_field(self) -> Optional[dict]:
        """Parse postProcessing/sample_z1p5/*/zPed.raw → dict, save wind_field.json."""
        try:
            import numpy as np
            from scipy.interpolate import griddata
        except ImportError:
            print("[CFD] scipy not installed — extract_wind_field() unavailable.")
            return None

        pp_dir = self.case_dir / "postProcessing" / "sample_z1p5"
        if not pp_dir.exists():
            return None
        time_dirs = sorted(
            [d for d in pp_dir.iterdir() if d.is_dir()],
            key=lambda d: float(d.name),
        )
        if not time_dirs:
            return None
        raw_file = next(
            (f for ext in ("raw", "xy") for f in [time_dirs[-1] / f"zPed.{ext}"] if f.exists()),
            None,
        )
        if raw_file is None:
            return None

        # Parse: columns are x_OF  y_OF  z_OF  Ux  Uy  Uz  (# comment lines skipped)
        xs, zs_3d, vxs, vzs = [], [], [], []
        for line in raw_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            of_x  = float(parts[0])
            of_y  = float(parts[1])   # streamwise → Three.js Z
            of_ux = float(parts[3])   # spanwise velocity → Three.js vx
            of_uy = float(parts[4])   # streamwise velocity → Three.js vz
            xs.append(of_x * SCENE_SCALE)
            zs_3d.append(of_y * SCENE_SCALE)
            vxs.append(of_ux)
            vzs.append(of_uy)

        if not xs:
            return None

        pts = np.array(list(zip(xs, zs_3d)))
        step = 2.5
        xi = np.arange(min(xs), max(xs) + step, step)
        zi = np.arange(min(zs_3d), max(zs_3d) + step, step)
        XI, ZI = np.meshgrid(xi, zi)
        grid_pts = np.column_stack([XI.ravel(), ZI.ravel()])

        ux_g = griddata(pts, vxs, grid_pts, method="linear", fill_value=0.0)
        uz_g = griddata(pts, vzs, grid_pts, method="linear", fill_value=0.0)

        result: dict = {
            "model": "rans_cfd",
            "grid": {
                "x_min": float(xi[0]),  "x_max": float(xi[-1]),
                "x_step": step, "nx": int(len(xi)),
                "z_min": float(zi[0]),  "z_max": float(zi[-1]),
                "z_step": step, "nz": int(len(zi)),
            },
            "ux": ux_g.tolist(),
            "uz": uz_g.tolist(),
        }
        wf = self.case_dir / "wind_field.json"
        wf.write_text(json.dumps(result))
        self._wind_field = result
        print(f"[CFD] Wind field saved to {wf}")
        return result

    def get_wind_velocity(self, x: float, z: float) -> tuple[float, float]:
        if self._wind_field is None:
            wf = self.case_dir / "wind_field.json"
            if wf.exists():
                self._wind_field = json.loads(wf.read_text())
            else:
                return ContinuityWindSolver().get_wind_velocity(x, z)
        return _bilinear_interpolate(self._wind_field, x, z)


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def _bilinear_interpolate(wind_field: dict, x: float, z: float) -> tuple[float, float]:
    g = wind_field["grid"]
    nx, nz = g["nx"], g["nz"]
    ux, uz = wind_field["ux"], wind_field["uz"]

    x = max(g["x_min"], min(g["x_max"], x))
    z = max(g["z_min"], min(g["z_max"], z))

    fx = (x - g["x_min"]) / g["x_step"]
    fz = (z - g["z_min"]) / g["z_step"]
    ix0 = max(0, min(nx - 2, int(fx)))
    iz0 = max(0, min(nz - 2, int(fz)))
    ix1, iz1 = ix0 + 1, iz0 + 1
    tx = fx - ix0
    tz = fz - iz0

    def idx(iz, ix): return iz * nx + ix

    vx = (
        (1 - tz) * ((1 - tx) * ux[idx(iz0, ix0)] + tx * ux[idx(iz0, ix1)]) +
        tz        * ((1 - tx) * ux[idx(iz1, ix0)] + tx * ux[idx(iz1, ix1)])
    )
    vz = (
        (1 - tz) * ((1 - tx) * uz[idx(iz0, ix0)] + tx * uz[idx(iz0, ix1)]) +
        tz        * ((1 - tx) * uz[idx(iz1, ix0)] + tx * uz[idx(iz1, ix1)])
    )
    return float(vx), float(vz)


# ---------------------------------------------------------------------------
# STL generation (ASCII)
# ---------------------------------------------------------------------------

def _face(nx, ny, nz, v0, v1, v2) -> str:
    return (
        f"  facet normal {nx} {ny} {nz}\n"
        f"    outer loop\n"
        f"      vertex {v0[0]} {v0[1]} {v0[2]}\n"
        f"      vertex {v1[0]} {v1[1]} {v1[2]}\n"
        f"      vertex {v2[0]} {v2[1]} {v2[2]}\n"
        f"    endloop\n"
        f"  endfacet"
    )


def _box_triangles(xn, xx, yn, yx, zn, zx) -> list[str]:
    """12 triangles for a box in OpenFOAM coords (Y=streamwise, Z=vertical)."""
    A, B, C, D = (xn,yn,zn), (xx,yn,zn), (xx,yx,zn), (xn,yx,zn)
    E, F, G, H = (xn,yn,zx), (xx,yn,zx), (xx,yx,zx), (xn,yx,zx)
    return [
        _face(0, 0, -1, A, D, C), _face(0, 0, -1, A, C, B),  # z=zn bottom
        _face(0, 0,  1, E, F, G), _face(0, 0,  1, E, G, H),  # z=zx top
        _face(0,-1,  0, A, B, F), _face(0,-1,  0, A, F, E),  # y=yn front
        _face(0, 1,  0, D, H, G), _face(0, 1,  0, D, G, C),  # y=yx back
        _face(-1,0,  0, A, E, H), _face(-1,0,  0, A, H, D),  # x=xn left
        _face( 1,0,  0, B, C, G), _face( 1,0,  0, B, G, F),  # x=xx right
    ]


def _buildings_to_stl(buildings: list[dict]) -> str:
    """Combine all buildings into one ASCII STL under a single solid 'buildings'."""
    lines = ["solid buildings"]
    for b in buildings:
        # Three.js → OpenFOAM coordinate mapping
        of_xn, of_xx = b["x_min"], b["x_max"]
        of_yn, of_yx = b["z_min"], b["z_max"]   # ThreeJS Z → OF Y
        of_zn, of_zx = 0.0, float(b["height"])   # ThreeJS Y → OF Z
        lines.extend(_box_triangles(of_xn, of_xx, of_yn, of_yx, of_zn, of_zx))
    lines.append("endsolid buildings")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenFOAM dictionary generators
# ---------------------------------------------------------------------------

def _foam_header(cls: str, obj: str) -> str:
    return (
        "FoamFile\n{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {cls};\n"
        f"    object      {obj};\n"
        "}\n"
    )


def _blockMeshDict(xmin, xmax, ymin, ymax, zmin, zmax, dx: float = 4.0) -> str:
    nx = max(1, round((xmax - xmin) / dx))
    ny = max(1, round((ymax - ymin) / dx))
    nz = max(1, round((zmax - zmin) / dx))
    return (
        _foam_header("dictionary", "blockMeshDict") + f"""
scale 1;

vertices
(
    ({xmin} {ymin} {zmin})  // 0
    ({xmax} {ymin} {zmin})  // 1
    ({xmax} {ymax} {zmin})  // 2
    ({xmin} {ymax} {zmin})  // 3
    ({xmin} {ymin} {zmax})  // 4
    ({xmax} {ymin} {zmax})  // 5
    ({xmax} {ymax} {zmax})  // 6
    ({xmin} {ymax} {zmax})  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces ( (0 1 5 4) );
    }}
    outlet
    {{
        type patch;
        faces ( (3 7 6 2) );
    }}
    ground
    {{
        type wall;
        faces ( (0 3 2 1) );
    }}
    top
    {{
        type patch;
        faces ( (4 5 6 7) );
    }}
    sides
    {{
        type patch;
        faces ( (0 4 7 3) (1 2 6 5) );
    }}
);
"""
    )


def _snappyHexMeshDict(buildings: list[dict], location_in_mesh=(0, -25, 5)) -> str:
    lx, ly, lz = location_in_mesh
    return (
        _foam_header("dictionary", "snappyHexMeshDict") + f"""
castellatedMesh  true;
snap             true;
addLayers        false;

geometry
{{
    buildings
    {{
        type triSurfaceMesh;
        file "buildings.stl";
    }}
}}

castellatedMeshControls
{{
    maxLocalCells        500000;
    maxGlobalCells       1000000;
    minRefinementCells   10;
    nCellsBetweenLevels  1;
    features             ();

    refinementSurfaces
    {{
        buildings
        {{
            level (1 2);
            patchInfo {{ type wall; inGroups (wall); }}
        }}
    }}

    refinementRegions {{}}

    resolveFeatureAngle 30;
    locationInMesh ({lx} {ly} {lz});
    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          30;
    nRelaxIter          5;
    nFeatureSnapIter    10;
    implicitFeatureSnap false;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    addLayers false;
    layers {{}}
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-30;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
    relaxed
    {{
        maxNonOrtho 75;
    }}
}}

mergeTolerance 1e-6;
"""
    )


def _controlDict() -> str:
    return (
        _foam_header("dictionary", "controlDict") + """
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         500;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
    sample_z1p5
    {
        type                surfaces;
        libs                (sampling);
        enabled             yes;
        writeControl        timeStep;
        writeInterval       500;
        surfaceFormat       raw;
        interpolationScheme cellPoint;
        fields              ( U );
        surfaces
        {
            zPed
            {
                type        cuttingPlane;
                planeType   pointAndNormal;
                pointAndNormalDict
                {
                    point  (0 0 1.5);
                    normal (0 0 1);
                }
                interpolate true;
            }
        }
    }
}
"""
    )


def _fvSchemes() -> str:
    return (
        _foam_header("dictionary", "fvSchemes") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwindV grad(U);
    div(phi,k)      Gauss linearUpwind grad(k);
    div(phi,epsilon) Gauss linearUpwind grad(epsilon);
    div((nuEff*dev(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""
    )


def _fvSolution() -> str:
    return (
        _foam_header("dictionary", "fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(k|epsilon|nut)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon)"   1e-4;
    }
}

relaxationFactors
{
    fields      { p 0.3; }
    equations   { U 0.7; k 0.7; epsilon 0.7; }
}
"""
    )


def _turbulence_init(U_ref: float, I: float = 0.05, L: float = 10.0):
    """Compute inlet k and epsilon from turbulence intensity I and length scale L."""
    k = 1.5 * (I * U_ref) ** 2
    Cmu = 0.09
    eps = (Cmu ** 0.75) * (k ** 1.5) / L
    return k, eps


def _0_U(U_ref: float) -> str:
    return (
        _foam_header("volVectorField", "U") + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 {U_ref} 0);

boundaryField
{{
    inlet   {{ type fixedValue;    value uniform (0 {U_ref} 0); }}
    outlet  {{ type zeroGradient; }}
    ground  {{ type noSlip; }}
    top     {{ type slip; }}
    sides   {{ type slip; }}
    buildings {{ type noSlip; }}
}}
"""
    )


def _0_p() -> str:
    return (
        _foam_header("volScalarField", "p") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet       { type zeroGradient; }
    outlet      { type fixedValue; value uniform 0; }
    ground      { type zeroGradient; }
    top         { type zeroGradient; }
    sides       { type zeroGradient; }
    buildings   { type zeroGradient; }
}
"""
    )


def _0_k(k_val: float) -> str:
    return (
        _foam_header("volScalarField", "k") + f"""
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {k_val:.6g};

boundaryField
{{
    inlet       {{ type fixedValue; value uniform {k_val:.6g}; }}
    outlet      {{ type zeroGradient; }}
    ground      {{ type kqRWallFunction; value uniform {k_val:.6g}; }}
    top         {{ type zeroGradient; }}
    sides       {{ type zeroGradient; }}
    buildings   {{ type kqRWallFunction; value uniform {k_val:.6g}; }}
}}
"""
    )


def _0_epsilon(eps_val: float) -> str:
    return (
        _foam_header("volScalarField", "epsilon") + f"""
dimensions      [0 2 -3 0 0 0 0];
internalField   uniform {eps_val:.6g};

boundaryField
{{
    inlet       {{ type fixedValue; value uniform {eps_val:.6g}; }}
    outlet      {{ type zeroGradient; }}
    ground      {{ type epsilonWallFunction; value uniform {eps_val:.6g}; }}
    top         {{ type zeroGradient; }}
    sides       {{ type zeroGradient; }}
    buildings   {{ type epsilonWallFunction; value uniform {eps_val:.6g}; }}
}}
"""
    )


def _0_nut() -> str:
    return (
        _foam_header("volScalarField", "nut") + """
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet       { type calculated; value uniform 0; }
    outlet      { type calculated; value uniform 0; }
    ground      { type nutkWallFunction; value uniform 0; }
    top         { type calculated; value uniform 0; }
    sides       { type calculated; value uniform 0; }
    buildings   { type nutkWallFunction; value uniform 0; }
}
"""
    )


def _transportProperties(nu: float = 1.5e-5) -> str:
    return (
        _foam_header("dictionary", "transportProperties") + f"""
transportModel  Newtonian;
nu              {nu};
"""
    )


def _turbulenceProperties() -> str:
    return (
        _foam_header("dictionary", "turbulenceProperties") + """
simulationType  RAS;

RAS
{
    RASModel    realizableKE;
    turbulence  on;
    printCoeffs on;
}
"""
    )
