import math
import random

import numpy as np
from numba import njit, prange


# @njit(parallel=True)
@njit
def update_grid_numba(
    state_grid_in, height_grid, density_grid, infection_time_in,
    recovery_time, P_h, cell_size_m,
    GREEN, ACTIVE, BURNED, DILUTED, RIVER, WATER
):
    grid_size = state_grid_in.shape[0]
    new_state_grid = np.copy(state_grid_in)
    new_infection_time = np.copy(infection_time_in)

    slope_factor = 0.078
    wind = 4.166
    theta_w = 7 * math.pi / 4
    c_1 = 0.045
    c_2 = 0.131

    NEIGHBORS_DATA = np.array([
        (-1,  0, 0), (-1,  1, 1), (0,  1, 2), (1,  1, 3),
        (1,  0, 4), (1, -1, 5), (0, -1, 6), (-1, -1, 7)
    ])

    THETA_D_VALUES = np.array([
        0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4,
        math.pi, 5 * math.pi / 4, 3 * math.pi / 2, 7 * math.pi / 4
    ])

    # for i in prange(grid_size):
    #     for j in prange(grid_size):
    for i in range(grid_size):
        for j in range(grid_size):
            current_state = state_grid_in[i, j]

            if current_state == GREEN:
                density = density_grid[i, j]
                if density < 0.25:
                    P_den = -0.6
                elif density < 0.5:
                    P_den = -0.3
                elif density < 0.75:
                    P_den = -0.1
                else:
                    P_den = 0.1

                P_veg = 0.0

                for k in range(8):
                    di, dj, theta_d_index = NEIGHBORS_DATA[k]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbor_state = state_grid_in[ni, nj]

                        if neighbor_state == ACTIVE:
                            current_height = height_grid[i, j]
                            neighbor_height = height_grid[ni, nj]

                            if k in [0, 2, 4, 6]:
                                distance = cell_size_m
                            else:
                                distance = cell_size_m * math.sqrt(2)

                            gradient = (neighbor_height - current_height) / distance
                            slope_angle = math.atan(gradient)
                            P_s = math.exp(slope_factor * slope_angle)

                            theta_d = THETA_D_VALUES[theta_d_index]
                            theta = abs(theta_w - theta_d)
                            P_w = math.exp(c_1 * wind) * math.exp(c_2 * wind * (math.cos(theta) - 1))

                            tau = random.random()
                            if k in [1, 3, 5, 7]:
                                tau = tau * math.sqrt(2)

                            P_burn = P_h * (1 + P_veg) * (1 + P_den) * P_w * P_s

                            if tau < P_burn:
                                new_state_grid[i, j] = ACTIVE
                                new_infection_time[i, j] = 0
                                break

            elif current_state == ACTIVE:
                recovery_step = 10 * (60 * (density_grid[i, j] ** 1.5)) / (1 + 0.2 * wind)

                new_infection_time[i, j] += 1
                if new_infection_time[i, j] >= recovery_step:
                    new_state_grid[i, j] = BURNED
                    new_infection_time[i, j] = 0

            elif current_state == RIVER or current_state == WATER or current_state == BURNED:
                pass

    return new_state_grid, new_infection_time


class GridUpdater:
    def __init__(self, params):
        self.params = params
        self.wind = 4.166

    def update_grid(self, grid, infection_time, get_neighbors, recovery_time, P_h, cell_size_m):
        grid_size = grid.shape[0]
        state_grid_in = np.zeros((grid_size, grid_size), dtype=np.int32)
        height_grid = np.zeros((grid_size, grid_size), dtype=np.float64)
        density_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

        for i in range(grid_size):
            for j in range(grid_size):
                cell = grid[i, j]
                state_grid_in[i, j] = cell.state
                height_grid[i, j] = cell.height
                density_grid[i, j] = cell.density

        new_state_grid, new_infection_time = update_grid_numba(
            state_grid_in, height_grid, density_grid, infection_time,
            recovery_time, P_h, cell_size_m,
            self.params["GREEN"], self.params["ACTIVE"], self.params["BURNED"],
            self.params["DILUTED"], self.params["RIVER"], self.params["WATER"],
        )

        for i in range(grid_size):
            for j in range(grid_size):
                grid[i, j].state = new_state_grid[i, j]

        return grid, new_infection_time

    @staticmethod
    @njit
    def active_function(t, n):
        if t < 0 or t > n or n == 0:
            return 0.0
        t_peak = n / 5
        if t <= t_peak:
            return t / t_peak if t_peak > 0 else 1.0
        return (1 - (t - t_peak) / (n - t_peak)) ** 2