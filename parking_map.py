import math
import heapq
import numpy as np

from config import (
    CENTER_WALL_X,
    PARKING_SPOT_WIDTH,
    RIGHT_WALL_X,
    ROW1_X_END,
    ROW1_X_START,
    ROW2_X_END,
    ROW2_X_START,
    ROW3_X_END,
    ROW3_X_START,
    ROW4_X_END,
    ROW4_X_START,
)


class UndergroundParkingMap:
    def __init__(self):
        ox, oy = [], []
        for x in np.arange(0, ROW3_X_END, 0.5):
            ox.append(x)
            oy.append(45.0)
        for x in np.arange(ROW4_X_START, RIGHT_WALL_X + 0.5, 0.5):
            ox.append(x)
            oy.append(45.0)
        for x in np.arange(0, RIGHT_WALL_X + 0.5, 0.5):
            ox.append(x)
            oy.append(0.0)
        for y in np.arange(0, 45.5, 0.5):
            ox.append(0.0)
            oy.append(y)
        for y in np.arange(0, 45.5, 0.5):
            ox.append(RIGHT_WALL_X)
            oy.append(y)
        for y in np.arange(4.5, 36.5, 0.5):
            ox.append(CENTER_WALL_X)
            oy.append(y)

        spot_width, start_y, end_y = PARKING_SPOT_WIDTH, 4.5, 36.5
        y_lines = np.arange(start_y, end_y + spot_width, spot_width)
        row1_x = np.arange(ROW1_X_START, ROW1_X_END + 1e-6, 0.5)
        row2_x = np.arange(ROW2_X_START, ROW2_X_END + 1e-6, 0.5)
        row3_x = np.arange(ROW3_X_START, ROW3_X_END + 1e-6, 0.5)
        row4_x = np.arange(ROW4_X_START, ROW4_X_END + 1e-6, 0.5)
        for y in y_lines:
            for x in row1_x:
                ox.append(x)
                oy.append(y)
            for x in row2_x:
                ox.append(x)
                oy.append(y)
            for x in row3_x:
                ox.append(x)
                oy.append(y)
            for x in row4_x:
                ox.append(x)
                oy.append(y)

        self.obstacles = list(zip(ox, oy))
        self.obs_arr = np.array(self.obstacles)
        self.dijkstra_grid = None
        self.grid_res = 1.0

    def get_do(self, x, y):
        dists = np.hypot(self.obs_arr[:, 0] - x, self.obs_arr[:, 1] - y)
        return np.min(dists)

    def precompute_2d_dijkstra(self, goal):
        print('>> 正在預先計算 2D Dijkstra 障礙物感知地圖...')
        w, h = int((RIGHT_WALL_X + 2.0) / self.grid_res), int(50 / self.grid_res)
        self.dijkstra_grid = np.full((w + 1, h + 1), float('inf'))

        gx, gy = int(goal.x / self.grid_res), int(goal.y / self.grid_res)
        if not (0 <= gx <= w and 0 <= gy <= h):
            return
        self.dijkstra_grid[gx, gy] = 0.0

        queue = [(0.0, gx, gy)]
        motions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

        while queue:
            cost, cx, cy = heapq.heappop(queue)
            for dx, dy in motions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx <= w and 0 <= ny <= h:
                    real_x, real_y = nx * self.grid_res, ny * self.grid_res
                    # Inflate obstacles in the Dijkstra grid so guidance stays away from walls.
                    if self.get_do(real_x, real_y) < 1.0:
                        continue

                    n_cost = cost + math.hypot(dx, dy) * self.grid_res
                    if n_cost < self.dijkstra_grid[nx, ny]:
                        self.dijkstra_grid[nx, ny] = n_cost
                        heapq.heappush(queue, (n_cost, nx, ny))
