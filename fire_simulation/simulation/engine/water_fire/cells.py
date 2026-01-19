class Cell:
    def __init__(self, state, height=0.0, density=1.0):
        self.state = state
        self.height = height
        self.density = density
        self.infection_time = 0

    def __repr__(self):
        return f"Cell(state={self.state}, height={self.height:.2f}, density={self.density:.2f})"

    @staticmethod
    def get_neighbors(grid, i, j):
        """グリッドと座標(i, j)から8近傍セルと方角を返す"""
        neighbors = []
        directions = [
            (-1, 0, "North"), (-1, 1, "North-East"), (0, 1, "East"), (1, 1, "South-East"),
            (1, 0, "South"), (1, -1, "South-West"), (0, -1, "West"), (-1, -1, "North-West")
        ]
        grid_size = grid.shape[0]
        for di, dj, direction in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                neighbors.append((grid[ni, nj], direction))
        return neighbors