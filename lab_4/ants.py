import numpy as np
from random import random
from tqdm import tqdm
import copy
import time

"""
coef

"a": ,
"b": ,
"evaporation": ,
"Q": ,
"ants_step": ,
"""

class AntSolver:
    def __init__(self, coef, TSP):
        self.startup_time = time.time()
        self.coef = coef
        self.check_path = TSP.check_path
        self.antilengths = np.power(np.divide(1,np.asarray(TSP.matrix)),self.coef["b"])
        self.coef["n_vertices"] = len(TSP.matrix)
        if not "ants_step" in self.coef.keys():
            self.coef["ants_step"] = 1
        
        self.pheromones = np.ones((self.coef["n_vertices"], self.coef["n_vertices"]))

        self.startup_time = time.time() - self.startup_time

    def func(self,path):
        return self.check_path(path.tolist())

    def reset(self):
        self.pheromones = np.ones((self.coef["n_vertices"], self.coef["n_vertices"]))

    def iter(self, force_debug = False):

        weights_memory = np.multiply(np.power(self.pheromones, self.coef["a"]),self.antilengths)
        delta_pheromones = np.zeros((self.coef["n_vertices"], self.coef["n_vertices"]))

        #run ants

        best_path = []
        best_path_length = 0
        paths = np.zeros((self.coef["n_vertices"],self.coef["n_vertices"]), dtype=int)
        for start_pos in range(0,self.coef["n_vertices"],self.coef["ants_step"]):

            paths[start_pos][0] = start_pos
            elimination_vector = np.ones((self.coef["n_vertices"]))
            elimination_vector[start_pos] = 0
            for i in range(self.coef["n_vertices"]-1):
                weights = np.multiply(weights_memory[paths[start_pos][i]], elimination_vector)
                weights /= np.sum(weights)

                w_sum = np.cumsum(weights)
                paths[start_pos][i + 1] = np.searchsorted(w_sum, np.random.rand())
                elimination_vector[paths[start_pos][i + 1]] = 0


        paths_lengths = list(map(self.func, paths))


        for k in range(self.coef["n_vertices"]):

            additive = np.array(self.coef["Q"] / paths_lengths[k])
            from_idx = paths[k][:-1]
            to_idx = paths[k][1:]

            delta_pheromones[from_idx, to_idx] += additive
            delta_pheromones[to_idx, from_idx] += additive

            delta_pheromones[paths[k][0], paths[k][-1]] += additive
            delta_pheromones[paths[k][-1], paths[k][0]] += additive

            if k == 0:
                best_path = paths[k]
                best_path_length = paths_lengths[k]
            else:
                if paths_lengths[k] < best_path_length:
                    best_path = paths[k]
                    best_path_length = paths_lengths[k]
            if force_debug: print(paths[k])
            if force_debug: print()
        if force_debug: print("---------")

        self.pheromones *= (1-self.coef["evaporation"])
        self.pheromones += np.asarray(delta_pheromones)
        return [best_path_length, best_path]

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

    def anisolve(self, TSP, iterations = 100, step = 5, minimum_frame_time = 0.5, force_debug = False):
        best_path = []
        best_length = 0
        for i in range(iterations//step):
            t = time.time()
            for j in range(step):
                result = self.iter()
                if i+j == 0:
                    print("startup")
                    best_path = result[1]
                    best_length = result[0]
                else:
                    if best_length > result[0]:
                        best_path = result[1]
                        best_length = result[0]
                if force_debug: print(np.round(self.pheromones,3).astype(float))
                if force_debug: print()
            TSP.draw_graph(path = best_path, matrix = self.pheromones, matrix_line_width=50)
            time.sleep(max(t + minimum_frame_time - time.time(), 0))

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="ants")
        best_path = []
        best_length = 0
        for i in iterator:
            result = self.iter()
            if best_length == 0:
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