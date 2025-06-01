from Backpack import Backpack
import numpy as np
from tqdm import tqdm
from copy import deepcopy

class GreedyBackpackSolver:
    def __init__(self, backpack: Backpack):
        self.backpack = backpack

    def reset(self):
        pass

    def solve(self, iterations = 0, progressbar = False):
        densities = self.backpack.objects/self.backpack.objects[:, -1].reshape(-1, 1)
        densities = 1/densities
        choice = np.zeros(self.backpack.objects.shape[0])
        iterator = range(self.backpack.objects.shape[0])
        if progressbar:
            iterator = tqdm(iterator)
        for i in iterator:
            completeness = self.backpack.fitness(choice)[:-1]/self.backpack.capacity
            worst_axis = np.argmax(completeness)
            best_object_for_worst_axis = np.argmax(densities[:, worst_axis])
            new_choice = deepcopy(choice)
            new_choice[best_object_for_worst_axis] = 1
            densities[best_object_for_worst_axis] = np.zeros_like(densities[best_object_for_worst_axis])
            if not self.backpack.fitness(new_choice)[-1] < 0:
                choice = new_choice
        return choice
