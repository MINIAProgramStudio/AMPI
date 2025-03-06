from random import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation, cm
"""
coef list
delta: ,
n_max: ,
alpha: ,
pop_size: ,
l_s: ,
l_es: ,
z_e: ,
z_o: ,
dim: ,
minmax:
"""
class BEEHive:
    def __init__(self, coef, func, seeking_min = False):
        self.c = coef
        self.func = func
        self.seeking_min = seeking_min
        self.best_pos = [random()*(self.c["minmax"][d][1]-self.c["minmax"][d][0])+self.c["minmax"][d][0] for d in range(self.c["dim"])]
        self.best_value = self.func(self.best_pos)
        self.bees = [[random()*(self.c["minmax"][d][1]-self.c["minmax"][d][0])+self.c["minmax"][d][0] for d in range(self.c["dim"])]
                     for _ in range(self.c["pop_size"])]
        self.iterations = 0

    def iter(self):
        self.iterations += 1
        self.bees.sort(key = self.func, reverse = not self.seeking_min)
        pop_best = self.func(self.bees[0])
        if pop_best > self.best_value and not self.seeking_min:
            self.best_pos = self.bees[0]
            self.best_value = pop_best
        elif pop_best < self.best_value and not self.seeking_min:
            self.best_pos = self.bees[0]
            self.best_value = pop_best

        for l in range(self.c["l_s"]):
            if l <= self.c["l_es"]:
                z = self.c["z_e"]
            else:
                z = self.c["z_o"]
            nu = self.c["n_max"]*self.c["alpha"]**self.iterations

            x_z = np.array([
                self.bees[l][d] + nu * self.c["delta"](self.c["minmax"][d][1] - self.c["minmax"][d][0]) * (-1 + 2*random())
                for d in range(self.c["dim"])
            ])
            x_z = np.maximum(self.c["minmax"][:][0], np.minimum(self.c["minmax"][:][1], x_z))
            x_z.sort(key = self.func, reverse = not self.seeking_min)
            z_best_pos = x_z[0]
            if self.func(z_best_pos) > self.func(self.bees[l]) and not self.seeking_min:
                self.bees[l] = z_best_pos
            elif self.func(z_best_pos) < self.func(self.bees[l]) and self.seeking_min:
                self.bees[l] = z_best_pos

        for l in range(self.c["l_s"], self.c["pop_size"]):
            self.bees[l] = [random()*(self.c["minmax"][d][1]-self.c["minmax"][d][0])+self.c["minmax"][d][0] for d in range(self.c["dim"])]

        def solve(itertaions = 100, progressbar = False):
            iterator = range(itertaions)
            if progressbar:
                iterator = tqdm(iterator, desc = "BEE")
            for _ in iterator:
                self.iter()
            self.bees.sort(key=self.func, reverse=not self.seeking_min)
            return (self.func(self.bees[0]), self.bees[0])