from copy import deepcopy

from Backpack import *
import numpy as np
from random import random
from tqdm import tqdm

"""
coef = {
    "pop_size": ,
    "children": ,
    "m_prob": ,
}

"""


class GeneticBackSolver:
    def __init__(self, backpack: Backpack, coef: dict):
        self.c = coef
        self.backpack = backpack
        self.reset()

    def reset(self):
        self.population = np.random.randint(0,2, (self.c["pop_size"], self.backpack.objects.shape[0]))
        self.best_pop = self.population[0]
        self.best_value = self.backpack.fitness(self.best_pop)
        self.sort_and_truncate()

    def sort_and_truncate(self):
        fitness_array = np.apply_along_axis(self.backpack.fitness, 1, self.population)
        sorted_indices = np.argsort(-fitness_array[:, -1])
        self.population = self.population[sorted_indices]
        self.population = self.population[: self.c["pop_size"], :]
        value = self.backpack.fitness(self.population[0])
        if value[-1] > self.best_value[-1]:
            self.best_pop = self.population[0]
            self.best_value = value

    def mutate(self):
        for pop in range(self.population.shape[0]):
            for gen in range(self.population.shape[1]):
                if random() < self.c["m_prob"]:
                    self.population[pop][gen] = 1 - self.population[pop][gen]

    def create_children(self):
        for i in range(self.c["children"]//2):
            pops = np.random.choice(self.population.shape[0], 2)
            pop_a = deepcopy(self.population[pops[0]])
            pop_b = deepcopy(self.population[pops[1]])
            for j in range(pop_a.shape[0]):
                if random() < 0.5:
                    pop_a[j], pop_b[j] = pop_b[j], pop_a[j]
            new_children = np.vstack([pop_a, pop_b])
            self.population = np.vstack([self.population, new_children])

    def iter(self):
        self.create_children()
        self.mutate()
        self.sort_and_truncate()

    def solve(self, iterations = 100, progressbar = False):
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator)
        for i in iterator:
            self.iter()
        return self.best_pop, self.best_value

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator)
        for i in iterator:
            self.iter()
            output.append(self.best_value)
        return self.best_pop, self.best_value, np.array(output)