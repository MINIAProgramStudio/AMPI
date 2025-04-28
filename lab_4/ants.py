import numpy as np
from random import random
from tqdm import tqdm
import copy
import time

"""
coef

"a": ,
"b": , # a + b = 1
"evaporation": ,
"Q": ,
"""

class AntSolver:
    def __init__(self, coef, TSP):
        self.startup_time = time.time()
        self.coef = coef
        self.func = TSP.check_path

        self.lengths = TSP.matrix
        self.antilengths = np.power(1/TSP.matrix,self.coef["b"])
        self.coef["n_vertices"] = len(TSP.matrix)
        self.coef["evaporation"] = (1 - self.coef["evaporation"]) ** self.coef["a"]

        # generate weights
        self.weights_memory = np.zeros((self.coef["n_vertices"], self.coef["n_vertices"]))
        for start_pos in range(self.coef["n_vertices"]):
            weights_for_starting_pos = np.zeros((self.coef["n_vertices"]))

            for end_pos in range(start_pos + 1, self.coef["n_vertices"]):
                weights_for_starting_pos[end_pos] = 0.01 ** self.coef["a"] * \
                                                    self.antilengths[start_pos][end_pos]
            self.weights_memory[start_pos] = weights_for_starting_pos
        for start_pos in range(self.coef["n_vertices"]):
            for end_pos in range(start_pos):
                self.weights_memory[start_pos][end_pos] = self.weights_memory[end_pos][start_pos]

            self.weights_memory[start_pos][start_pos] = 0

        self.startup_time = time.time() - self.startup_time

    def reset(self):

        # generate weights
        self.weights_memory = np.zeros((self.coef["n_vertices"], self.coef["n_vertices"]))
        for start_pos in range(self.coef["n_vertices"]):
            weights_for_starting_pos = np.zeros((self.coef["n_vertices"]))

            for end_pos in range(start_pos + 1, self.coef["n_vertices"]):
                weights_for_starting_pos[end_pos] = 0.01 ** self.coef["a"] * \
                                                    self.antilengths[start_pos][end_pos]
            self.weights_memory[start_pos] = weights_for_starting_pos
        for start_pos in range(self.coef["n_vertices"]):
            for end_pos in range(start_pos):
                self.weights_memory[start_pos][end_pos] = self.weights_memory[end_pos][start_pos]

            self.weights_memory[start_pos][start_pos] = 0

    def iter(self):
        #balance weights
        for start_pos in range(self.coef["n_vertices"]):
            self.weights_memory[start_pos] /= np.sum(self.weights_memory[start_pos])
        weights_additive = np.zeros((self.coef["n_vertices"], self.coef["n_vertices"]))
        #run ants
        best_path = []
        best_path_length = 0
        for start_pos in range(self.coef["n_vertices"]):
            path = np.zeros((self.coef["n_vertices"])).astype(int)
            path[0] = start_pos
            path_length = 0
            for i in range(self.coef["n_vertices"]-1):
                weights = copy.deepcopy(self.weights_memory[path[i]])
                eliminated_weights = 0
                for j in range(self.coef["n_vertices"]):
                    if j in path:
                        eliminated_weights += weights[j]
                        weights[j] = 0
                weights = weights / (1-eliminated_weights)

                r = random()
                selected_path = 0
                for j in range(self.coef["n_vertices"]):
                    r -= weights[selected_path]
                    if r < 0:
                        break
                    selected_path += 1
                if selected_path >= self.coef["n_vertices"]:
                    selected_path = self.coef["n_vertices"]-1
                path[i+1] = selected_path
                path_length += self.lengths[path[i]][selected_path]
                weights_additive[path[i]][selected_path] += (self.coef["Q"]/path_length)**self.coef["a"]
                weights_additive[selected_path][path[i]] += (self.coef["Q"]/path_length) ** self.coef["a"]
            if start_pos == 0:
                best_path = path
                best_path_length = path_length
            else:
                path_length = path_length
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

        self.weights_memory *= self.coef["evaporation"]
        self.weights_memory += weights_additive
        return [best_path_length, best_path.tolist()]

    def solve(self, iterations = 100, progressbar = False):
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="ants")
        best_path = []
        best_length = 0
        for _ in iterator:
            result = self.iter()
            if _ == 0:
                best_path = result[1]
                best_length = result[0]
            else:
                if best_length > result[0]:
                    best_path = result[1]
                    best_length = result[0]
        return [best_length, best_path]

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="ants")
        best_path = []
        best_length = 0
        for i in iterator:
            result = self.iter()
            if i == 0:
                best_path = result[1]
                best_length = result[0]
            else:
                if best_length > result[0]:
                    best_path = result[1]
                    best_length = result[0]
            output.append(best_length)
        return (best_length, best_path, output)

    def solve_seconds(self, seconds = 10):
        output = []
        start = time.time()-self.startup_time
        best_path = []
        best_length = 0
        while start + seconds > time.time():
            result = self.iter()
            if best_length == 0:
                best_path = result[1]
                best_length = result[0]
            else:
                if best_length > result[0]:
                    best_path = result[1]
                    best_length = result[0]
            output.append([time.time()-start,best_length])
        return (best_length, best_path, output)