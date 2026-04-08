import math

from config import PARAMS


class State:
    def __init__(self, x, y, theta, gear=1, steer=0.0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.gear = gear
        self.steer = steer
        self.g_cost = 0.0
        self.f_cost = 0.0
        self.parent = parent

    def get_corners(self):
        L = PARAMS['vehicle_L']
        W = PARAMS['vehicle_W']
        WB = PARAMS['wheelbase']

        # State (x, y) is rear-axle center in the bicycle model.
        rear_overhang = (L - WB) / 2.0
        front_edge = WB + rear_overhang
        rear_edge = -rear_overhang

        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)
        offsets = [
            (front_edge, W / 2),
            (front_edge, -W / 2),
            (rear_edge, -W / 2),
            (rear_edge, W / 2),
        ]
        return [
            (self.x + dx * cos_t - dy * sin_t, self.y + dx * sin_t + dy * cos_t)
            for dx, dy in offsets
        ]
