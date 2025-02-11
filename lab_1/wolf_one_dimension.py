from random import random

class OneDimentionalWolf:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

class OneDimentionalPackOfWolfs:
    def __init__(self, x_min, x_max, func, n_of_wolfs, speed_func):
        self.x_min = x_min
        self.x_max = x_max
        self.func = func
        if n_of_wolfs >= 2:
            self.wolfs = [OneDimentionalWolf]