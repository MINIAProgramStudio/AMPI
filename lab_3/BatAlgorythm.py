from random import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation, cm
from time import time

"""
coef list
"r0": ,
"A0": ,
"alpha": ,
"delta": ,
"gamma": ,
"freq_min": ,
"freq_max": ,


"pop_size": ,
"dim": ,
"pos_min": ,
"pos_max": 
"""




class BatSwarm:
    def __init__(self, coef, func, seeking_min=False):
        self.coef = coef
        self.func = func
        self.seeking_min = seeking_min

        self.population = [[
            np.array([random() * (self.coef["pos_max"][d] - self.coef["pos_min"][d]) + self.coef["pos_min"][d] for d in range(self.coef["dim"])]),
            np.zeros(self.coef["dim"]),
            self.coef["freq_min"]]
            for _ in range(self.coef["pop_size"])]
        self.sort()

        self.iteration_count = 0
        self.A = self.coef["A0"]
        self.r = self.coef["r0"]

    def reset(self):
        self.population = [[
            np.array([random() * (self.coef["pos_max"][d] - self.coef["pos_min"][d]) + self.coef["pos_min"][d] for d in
                      range(self.coef["dim"])]),
            np.zeros(self.coef["dim"]),
            self.coef["freq_min"]]
            for _ in range(self.coef["pop_size"])]
        self.iteration_count = 0
        self.A = self.coef["A0"]
        self.r = self.coef["r0"]

    def sort(self):
        def sort_func(bat):
            return self.func(bat[0])

        self.population.sort(key = sort_func, reverse = not self.seeking_min)
        return self.population


    def iterate(self):
        self.iteration_count += 1
        for k in range(self.coef["pop_size"]):
            self.population[k][2] = self.coef["freq_min"] + (self.coef["freq_max"]-self.coef["freq_min"])*random()
            self.population[k][1] += (self.population[0][0]-self.population[k][0])*self.population[k][2]
            self.population[k][0] += self.population[k][1]
            if random() < self.r:
                x_current = self.population[0][0] + self.coef["delta"]*(np.array(self.coef["pos_max"])-np.array(self.coef["pos_min"]))*(np.array([-1 + 2*random()]*self.coef["dim"]))
                x_current = np.minimum(self.coef["pos_max"], np.maximum(self.coef["pos_min"], x_current))
                if self.func(x_current) <= self.func(self.population[k][0]) and random()<self.A:
                    self.population[k][0] = x_current
                    self.A = self.coef["alpha"]*self.A
                    self.r = self.coef["r0"]*(1-np.exp(-self.coef["gamma"]*self.iteration_count))
                if self.func(x_current) <= self.func(self.population[0][0]):
                    self.population[0][0] = x_current
        self.sort()

    def solve(self, iterations = 100, progressbar = False):
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc = "BatSwarm")
        for _ in iterator:
            self.iterate()
        return (self.func(self.population[0][0]), self.population[0][0])

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="BatSwarm")
        for _ in iterator:
            self.iterate()
            if output:
                output.append(min(self.func(self.population[0][0][0]), output[-1]))
            else:
                output.append(self.func(self.population[0][0][0]))
        output.append(min(self.func(self.population[0][0][0]), output[-1]))
        return (self.func(self.population[0][0]), self.population[0][0], output)

