import numpy as np
from random import random

"""
coef

n_vertices: ,
pop_size: ,
m_switch_prob: ,
"""

class GTSP:
    def __init__(self, coef, path_eval_func, seeking_min = True):
        self.func = path_eval_func
        self.seeking_min = seeking_min
        self.coef = coef

        self.population = [np.shuffle(np.arrange(0,self.coef["n_vertices"])) for _ in range(self.coef["pop_size"])]

    def sort(self):
        self.population.sort(key = self.func, reverse = not self.seeking_min)

    def mutate_switch(self):
        for i in range(self.coef["pop_size"]):
            if random() < self.coef["m_switch_prob"]:
                a, b = int(random()*self.coef["pop_size"]), int(random()*self.coef["pop_size"])
                self.population[i][a], self.population[i][b] = self.population[i][b], self.population[i][a]
