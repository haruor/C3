import math

# --- Simulation Parameters ---
GRID_SIZE = 200
CELL_SIZE_M = 10

P_H = 0.010
RECOVERY_TIME = 150

# --- Physics / Environmental Constants ---
SLOPE_FACTOR = 0.078

WIND_SPEED = 4.5
THETA_W = 3 * math.pi / 4

C_1 = 0.09
C_2 = 0.262

# --- Interaction / Water Control ---
ACTIVE_THRESHOLD = 50

WATER_ON_ACTIVE_DURATION = 2
WATER_ON_GREEN_DURATION = 2
WATER_ON_BURNED_DURATION = 2

MAX_WATER_CELLS_PER_DRAG_STEP = 9

# --- Animation / View (現状Django版では未使用でも保持) ---
ANIMATION_FPS = 30
VIEW_WINDOW_SIZE = 45