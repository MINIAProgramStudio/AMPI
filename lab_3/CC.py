from random import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation, cm
from time import time
"""
coef list
"p_detect": ,
"delta": ,


"pop_size": ,
"dim": ,
"pos_min": ,
"pos_max": 
"""
class CC:
    def __init__(self, coef, func, seeking_min = False):
        self.coef = coef
        self.func = func
        self.seeking_min = seeking_min
        self.population = [np.array([random()*(self.coef["pos_max"][d] - self.coef["pos_min"][d]) + self.coef["pos_min"][d] for d in range(self.coef["dim"])]) for _ in range(self.coef["pop_size"])]
        self.population.sort(key = self.func, reverse = self.seeking_min)
        self.best_pos = self.population[-1]
        self.best_val = self.func(self.best_pos)

    def reset(self):
        self.population = [np.array(
            [random() * (self.coef["pos_max"][d] - self.coef["pos_min"][d]) + self.coef["pos_min"][d] for d in
             range(self.coef["dim"])]) for _ in range(self.coef["pop_size"])]
        self.population.sort(key=self.func, reverse=self.seeking_min)
        self.best_pos = self.population[-1]
        self.best_val = self.func(self.best_pos)

    def iterate(self):
        k = int((self.coef["pop_size"])*random())
        x_current = self.population[k] + self.coef["delta"] * np.array([(random()*2 - 1)*(self.coef["pos_max"][d] - self.coef["pos_min"][d]) for d in range(self.coef["dim"])])
        x_current = np.minimum(self.coef["pos_max"], np.maximum(self.coef["pos_min"], x_current))
        x_value = self.func(x_current)
        if self.seeking_min and x_value < self.best_val:
            self.best_val = x_value
            self.best_pos = x_current
            self.population[k] = self.best_pos
            self.population.sort(key = self.func, reverse = self.seeking_min)
        elif (not self.seeking_min) and x_value > self.best_val:
            self.best_val = x_value
            self.best_pos = x_current
            self.population[k] = self.best_pos
            self.population.sort(key = self.func, reverse = self.seeking_min)
        if random() < self.coef["delta"]:
            self.population[0] += self.coef["delta"] * np.array([(random()*2 - 1)*(self.coef["pos_max"][d] - self.coef["pos_min"][d]) for d in range(self.coef["dim"])])
            self.population[0] = np.minimum(self.coef["pos_max"], np.maximum(self.coef["pos_min"], self.population[0]))
            self.population.sort(key = self.func, reverse = self.seeking_min)

    def solve(self, iterations = 100, progressbar = False):
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc = "СС")
        for _ in iterator:
            self.iterate()
        self.population.sort(key = self.func, reverse = self.seeking_min)
        return (self.best_val, self.best_pos)

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="СС")
        for _ in iterator:
            self.iterate()
            output.append(self.best_val)
        output.append(self.best_val)
        return (self.best_val, self.best_pos, output)

