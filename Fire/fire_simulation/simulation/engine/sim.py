from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import random
import math
import io
import csv  # ★追加

import matplotlib
matplotlib.use("Agg")  # Webサーバ上でGUI不要
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # すでにあるはず
from matplotlib.patches import Patch  # 追加

from .water_fire.cells import Cell  # noqa: E402
from .water_fire.update_grid_numba import GridUpdater  # noqa: E402
from simulation import config as app_config


# 状態定数（元コード準拠）
GREEN, ACTIVE, BURNED, DILUTED, RIVER, WATER = 0, 1, 2, 3, 4, 5

# 元コードの10色 + DILUTED(追加) = 11色cmap（index: 0-10）
# 0-3: GREEN(密度4段階)
# 4-6: ACTIVE(燃焼強度3段階)
# 7  : BURNED
# 8  : RIVER
# 9  : WATER
# 10 : DILUTED（追加）
FIRE_CMAP_12 = ListedColormap([
    "#e0ffe0", "#80ff80", "#00cc44", "#006622",
    "#8B0000", "#DC143C", "#FF5050",
    "#646464",
    "deepskyblue",
    "cyan",
    "#A0A0A0",
    "#FFD54F",  # ★ index 11 OTHER（燃えないところ）: 水色→黄色に変更
])


@dataclass
class SimConfig:
    # 既存の描画系はそのまま
    frame_dpi: int = 120

    # config.py と合わせる
    cell_size_m: float = app_config.CELL_SIZE_M
    P_h: float = app_config.P_H
    recovery_time: int = app_config.RECOVERY_TIME

    slope_factor: float = app_config.SLOPE_FACTOR
    wind_speed: float = app_config.WIND_SPEED
    theta_w: float = app_config.THETA_W
    c1: float = app_config.C_1
    c2: float = app_config.C_2

    active_threshold: int = app_config.ACTIVE_THRESHOLD
    max_water_cells_per_drag_step: int = app_config.MAX_WATER_CELLS_PER_DRAG_STEP

    water_on_active_duration: int = app_config.WATER_ON_ACTIVE_DURATION
    water_on_green_duration: int = app_config.WATER_ON_GREEN_DURATION
    water_on_burned_duration: int = app_config.WATER_ON_BURNED_DURATION

    # 既存に river_density_threshold 等がある場合は、いったん現状値維持（後で整合）
    river_density_threshold: float = 0.1

    # ★表示用：正規化植生(0..1)で緑の濃さを分ける閾値（3つ＝4段階）
    # 例：0..0.25..0.5..0.75..1.0
    green_density_bins: tuple[float, float, float] = (0.25, 0.50, 0.75)

    # ★表示用：この値以下は「燃えない」扱いで黄色(OTHER=11)にする
    # （不要なら 0.0 にする）
    nonburn_density_threshold: float = 0.0


@dataclass
class SimState:
    elev_path: Path
    vege_path: Path
    ignition_path: Optional[Path] = None  # ★追加（任意）

    t: int = 0
    is_paused: bool = False
    water_mode: bool = False

    # 消火用の水量(L)
    water_remaining_l: int = 9500 * 3  # 28500L

    # ★救援（補給）のクールダウン：次に使用できるt
    relief_next_available_t: int = 0

    # エンジン内部状態（セッション保存したいが、まずはメモリに載せる）
    initialized: bool = False


class Simulator:
    """
    最小実装：
    - CSV(標高/植生) → height_grid / density_grid
    - Cellグリッド生成
    - 1 step更新
    - fire/elev をPNGで返す
    """

    RELIEF_COOLDOWN_STEPS = 50
    RELIEF_AMOUNTS_L = (9500, 11300)

    # --- 座標変換定数 (ForestFire_Auto_numba.py 準拠) ---
    GRID_ORIGIN_X = 521863.0
    GRID_ORIGIN_Y = 3383228.0
    GRID_RES = 44.835

    # --- Ignition 時間変換 (ForestFire_Auto_numba.py 準拠: 6秒=1step) ---
    IGNITION_SECONDS_PER_STEP = 6.0

    def __init__(self, state: SimState, config: Optional[SimConfig] = None) -> None:
        self.state = state
        self.config = config or SimConfig()

        # ★追加：動的着火イベント {time_step: [(r,c), ...]}
        self.ignition_events: dict[int, list[tuple[int, int]]] = {}

        # コア状態（初期化後に埋まる）
        self.grid: Optional[np.ndarray] = None  # dtype=object Cell
        self.state_grid: Optional[np.ndarray] = None  # int
        self.height_grid: Optional[np.ndarray] = None  # float
        self.density_grid: Optional[np.ndarray] = None  # float
        self.infection_time: Optional[np.ndarray] = None  # int

        # ★追加：水の残り時間（WATERセルが灰になるまで）
        self.water_time_left: Optional[np.ndarray] = None  # int

        # ★追加：水が切れたあとに戻すセル状態
        self.water_return_state: Optional[np.ndarray] = None  # int

        self.grid_updater = GridUpdater(
            {
                "GREEN": GREEN,
                "ACTIVE": ACTIVE,
                "BURNED": BURNED,
                "DILUTED": DILUTED,
                "RIVER": RIVER,
                "WATER": WATER,
            }
        )

        # ★ここがズレの原因だったので修正：
        # 以前の固定値（0.058, 217, 10）は使わず、config.pyと同じ値を使う
        self.P_h = float(self.config.P_h)                  # = app_config.P_H (=0.010)
        self.recovery_time = int(self.config.recovery_time) # = app_config.RECOVERY_TIME (=150)
        self.cell_size_m = float(self.config.cell_size_m)   # = app_config.CELL_SIZE_M (=10)

        # 12色cmapを使用
        self.cmap = FIRE_CMAP_12

    def _load_ignition_data(self, filepath: Path, grid_size: int) -> None:
        """
        ignition_synced_wide.csv を読み込み、タイムステップごとの着火点リストを作る（UTM前提）。
        self.ignition_events = { step: [(r,c), ...] }
        """
        self.ignition_events = {}

        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    elapsed_sec = float(row.get("Elapsed_Sec", ""))
                except ValueError:
                    continue

                step = int(elapsed_sec / float(self.IGNITION_SECONDS_PER_STEP))

                pts: list[tuple[int, int]] = []
                k = 1
                while True:
                    xk = f"Ign{k}_X"
                    yk = f"Ign{k}_Y"
                    if xk not in row or yk not in row:
                        break

                    sx = row.get(xk) or ""
                    sy = row.get(yk) or ""
                    if sx.strip() and sy.strip():
                        try:
                            utm_x = float(sx)
                            utm_y = float(sy)

                            c = int((utm_x - float(self.GRID_ORIGIN_X)) / float(self.GRID_RES))
                            r = int((float(self.GRID_ORIGIN_Y) - utm_y) / float(self.GRID_RES))

                            if 0 <= r < grid_size and 0 <= c < grid_size:
                                pts.append((r, c))
                        except ValueError:
                            pass

                    k += 1

                if pts:
                    self.ignition_events.setdefault(step, []).extend(pts)

    def _ensure_initialized(self) -> None:
        if self.state.initialized and self.grid is not None:
            return

        height = np.loadtxt(self.state.elev_path, delimiter=",", dtype=np.float64)
        veg = np.loadtxt(self.state.vege_path, delimiter=",", dtype=np.float64)

        # ★vegetation CSV は min-max で 0..1 正規化（最小=0, 最大=1）
        veg_raw = veg.astype(np.float64)
        vmin = float(np.min(veg_raw))
        vmax = float(np.max(veg_raw))
        if vmax > vmin:
            veg_value = (veg_raw - vmin) / (vmax - vmin)
        else:
            veg_value = np.zeros_like(veg_raw, dtype=np.float64)

        grid_size = int(height.shape[0])

        # ★グリッド生成（ここが欠けていた）
        grid = np.empty((grid_size, grid_size), dtype=object)
        state_grid = np.zeros((grid_size, grid_size), dtype=np.int32)

        for i in range(grid_size):
            for j in range(grid_size):
                d = float(veg_value[i, j])
                st = RIVER if d < float(self.config.river_density_threshold) else GREEN
                state_grid[i, j] = st
                grid[i, j] = Cell(state=st, height=float(height[i, j]), density=d)

        infection_time = np.zeros((grid_size, grid_size), dtype=np.int32)

        # ★水関連の初期化
        water_time_left = np.zeros((grid_size, grid_size), dtype=np.int32)
        water_return_state = state_grid.copy()

        # ★Ignition CSV読み込み（任意）
        self.ignition_events = {}
        if self.state.ignition_path is not None:
            try:
                self._load_ignition_data(self.state.ignition_path, grid_size=grid_size)
            except Exception:
                self.ignition_events = {}

        # ★初期着火：Ignition CSV が無い（または読めずに空）なら中央着火
        if not self.ignition_events:
            ci = grid_size // 2
            cj = grid_size // 2
            state_grid[ci, cj] = ACTIVE
            grid[ci, cj].state = ACTIVE

        self.grid = grid
        self.state_grid = state_grid
        self.height_grid = height
        self.density_grid = veg_value
        self.infection_time = infection_time
        self.water_time_left = water_time_left
        self.water_return_state = water_return_state
        self.state.initialized = True

    def toggle_pause(self) -> bool:
        self.state.is_paused = not self.state.is_paused
        return self.state.is_paused

    def toggle_water(self) -> bool:
        self.state.water_mode = not self.state.water_mode
        return self.state.water_mode

    def step(self, n: int = 1) -> None:
        self._ensure_initialized()
        if self.state.is_paused:
            return
        assert self.grid is not None
        assert self.state_grid is not None
        assert self.infection_time is not None

        for _ in range(max(1, int(n))):
            # ★動的着火（現在stepに対応する点をACTIVE化）
            if self.ignition_events:
                pts = self.ignition_events.get(int(self.state.t), [])
                for (r, c) in pts:
                    st = int(self.grid[r, c].state)
                    if st in (RIVER, BURNED, WATER):
                        continue
                    if st != ACTIVE:
                        self.grid[r, c].state = ACTIVE
                        self.state_grid[r, c] = ACTIVE
                        self.infection_time[r, c] = 0

            # ★WATERは再燃しない保証：update_gridの前にマスク
            water_mask_before = np.zeros((self.grid.shape[0], self.grid.shape[1]), dtype=bool)
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if self.grid[i, j].state == WATER:
                        water_mask_before[i, j] = True

            self.grid, self.infection_time = self.grid_updater.update_grid(
                self.grid,
                self.infection_time,
                Cell.get_neighbors,
                self.recovery_time,
                self.P_h,
                self.cell_size_m,
            )

            # ★WATERはupdate_gridで変化させない（再燃防止）
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    if water_mask_before[i, j]:
                        self.grid[i, j].state = WATER

            # ★WATER -> 時間経過で復帰
            grid_size = self.grid.shape[0]
            for i in range(grid_size):
                for j in range(grid_size):
                    if self.grid[i, j].state == WATER:
                        self.water_time_left[i, j] = max(0, int(self.water_time_left[i, j]) - 1)
                        if self.water_time_left[i, j] <= 0:
                            self.grid[i, j].state = int(self.water_return_state[i, j])

            # state_gridを同期（簡易）
            for i in range(grid_size):
                for j in range(grid_size):
                    self.state_grid[i, j] = self.grid[i, j].state

            self.state.t += 1

    def _render_index_grid(self) -> np.ndarray:
        """
        state_grid(0..5) を描画用indexに変換する。
        入力の vegetation CSV は正規化済み(0..1)を想定。
        - GREEN: density(0..1) を 0..3 に段階化（緑4色）
        - 低密度（<=nonburn_density_threshold）は OTHER(11)=黄色
        """
        assert self.state_grid is not None
        assert self.density_grid is not None
        assert self.infection_time is not None

        idx = np.empty_like(self.state_grid, dtype=np.int32)
        idx[:, :] = 7  # デフォルト（BURNED）

        sg = self.state_grid

        # --- GREEN: 正規化密度で色を段階化 ---
        mask_g = sg == GREEN
        if np.any(mask_g):
            d = self.density_grid[mask_g].astype(np.float64)

            # 非燃焼（黄色）
            other = d <= float(self.config.nonburn_density_threshold)

            out = np.empty(d.shape, dtype=np.int32)

            # 緑の段階（0..3）
            bins = np.array(self.config.green_density_bins, dtype=np.float64)
            levels = np.digitize(d, bins, right=True).astype(np.int32)
            levels = np.clip(levels, 0, 3)

            out[:] = levels
            out[other] = 11  # OTHER（黄色）

            idx[mask_g] = out

        # ACTIVE -> 4..6 (infection_time 3段階)
        mask_a = sg == ACTIVE
        if np.any(mask_a):
            t = self.infection_time[mask_a].astype(np.float64)
            tmax = max(float(np.max(t)), 1.0)
            level = np.clip((t / tmax * 3).astype(np.int32), 0, 2)
            idx[mask_a] = 4 + level

        idx[sg == BURNED] = 7
        idx[sg == RIVER] = 8
        idx[sg == WATER] = 9
        idx[sg == DILUTED] = 10

        return idx

    @staticmethod
    def _transform_for_display(a: np.ndarray) -> np.ndarray:
        """
        表示変換なし（転置しない）。
        """
        return a

    def render_fire_png(self) -> bytes:
        self._ensure_initialized()

        fig, ax = plt.subplots(figsize=(5, 5), dpi=self.config.frame_dpi)

        render_idx = self._render_index_grid()
        render_idx = self._transform_for_display(render_idx)  # ★統一

        ax.imshow(render_idx, cmap=self.cmap, interpolation="nearest", vmin=0, vmax=11)
        ax.set_title(f"Fire Spread at Time: {self.state.t}")
        ax.axis("off")

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    def render_elev_png(self) -> bytes:
        self._ensure_initialized()
        assert self.height_grid is not None

        fig, ax = plt.subplots(figsize=(5, 5), dpi=self.config.frame_dpi)

        elev = self._transform_for_display(self.height_grid)  # ★統一
        im = ax.imshow(elev, cmap="terrain")

        ax.set_title("Elevation Heatmap (m)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

    WATER_COST_PER_CELL_L = 897

    def place_water(self, i: int, j: int, radius: int = 1) -> int:
        """
        (i,j)中心の(2r+1)x(2r+1)にWATERを置く。
        水量制限：1セル=897L消費。残量が無ければ置けない。
        戻り値は実際に置けたセル数。
        """
        self._ensure_initialized()
        assert self.grid is not None
        assert self.state_grid is not None
        assert self.water_time_left is not None
        assert self.water_return_state is not None

        n = int(self.grid.shape[0])
        if not (0 <= i < n and 0 <= j < n):
            return 0

        # 残量チェック
        max_cells = int(self.state.water_remaining_l // self.WATER_COST_PER_CELL_L)
        if max_cells <= 0:
            return 0

        r = max(0, int(radius))

        placed = 0
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if placed >= max_cells:
                    break

                ni = i + di
                nj = j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    # 川(RIVER)は置かない
                    if self.state_grid[ni, nj] == RIVER:
                        continue
                    # すでにWATERなら追加消費しない
                    if self.state_grid[ni, nj] == WATER:
                        continue

                    # ★水を置く前の状態を見て、復帰先を決める
                    prev_state = int(self.state_grid[ni, nj])
                    if prev_state == ACTIVE:
                        return_state = BURNED  # 鎮火後は灰
                    else:
                        return_state = prev_state  # 燃えてない場所は元に戻す（GREEN等）

                    self.water_return_state[ni, nj] = return_state

                    self.state_grid[ni, nj] = WATER
                    self.grid[ni, nj].state = WATER

                    dur = self._calc_extinguish_duration_steps(self.grid[ni, nj].density)
                    self.water_time_left[ni, nj] = dur

                    placed += 1

            if placed >= max_cells:
                break

        # 消費
        self.state.water_remaining_l -= placed * self.WATER_COST_PER_CELL_L
        if self.state.water_remaining_l < 0:
            self.state.water_remaining_l = 0

        return placed

    def can_relief(self) -> bool:
        return int(self.state.t) >= int(self.state.relief_next_available_t)

    def relief(self) -> dict:
        """
        救援：9500L or 11300L をランダム補給。
        使用後は50ステップ後まで使用不可。
        タンク上限なし（無限に貯められる）。
        """
        if not self.can_relief():
            remain = max(0, int(self.state.relief_next_available_t - self.state.t))
            return {"ok": False, "reason": "cooldown", "cooldown_remaining_steps": remain}

        add = int(random.choice(self.RELIEF_AMOUNTS_L))
        self.state.water_remaining_l = int(self.state.water_remaining_l) + add
        self.state.relief_next_available_t = int(self.state.t) + int(self.RELIEF_COOLDOWN_STEPS)

        return {
            "ok": True,
            "added_l": add,
            "water_remaining_l": int(self.state.water_remaining_l),
            "cooldown_remaining_steps": int(self.RELIEF_COOLDOWN_STEPS),
            "next_available_t": int(self.state.relief_next_available_t),
        }

    @staticmethod
    def _calc_extinguish_duration_steps(density: float) -> int:
        # densityが0..1を想定。念のため範囲外はクリップ
        d = float(np.clip(density, 0.0, 1.0))
        return int(math.ceil(34.0 + 33.0 * d))