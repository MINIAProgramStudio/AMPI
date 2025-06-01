from Backpack import Backpack
import numpy as np
from tqdm import tqdm

class GreedyBackpackSolver:
    def __init__(self, backpack: Backpack):
        self.backpack = backpack

    def reset(self):
        pass

    def solve(self, iterations = 0, progressbar = False):
        densities = self.backpack.objects/self.backpack.objects[:, -1]
        iterator = range(self.backpack.objects.shape[0])
        choice = np.zeros(self.backpack.objects.shape[0])
        if progressbar:
            iterator = tqdm(iterator)
        for i in iterator:
            completeness = self.backpack.fitness(choice)/self.backpack.capacity
            worst_axis = np.argmax(completeness)
            best_axis = np.argmin(completeness)
            