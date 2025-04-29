import numpy as np
from random import random
from tqdm import tqdm
import copy
import time
import cupy as cp

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
        self.func = TSP.check_path
        self.antilengths = cp.power(cp.divide(1,cp.asarray(TSP.matrix, dtype=cp.float16)),self.coef["b"])
        self.coef["n_vertices"] = len(TSP.matrix)
        if not "ants_step" in self.coef.keys():
            self.coef["ants_step"] = 1
        
        self.pheromones = cp.ones((self.coef["n_vertices"], self.coef["n_vertices"]),dtype=cp.float16)

        self.startup_time = time.time() - self.startup_time

    def reset(self):
        self.pheromones = np.ones((self.coef["n_vertices"], self.coef["n_vertices"]))

    def iter(self, force_debug = False):
        weights_memory = cp.asnumpy(cp.multiply(cp.power(self.pheromones, self.coef["a"]),self.antilengths))
        delta_pheromones = np.zeros((self.coef["n_vertices"], self.coef["n_vertices"]),dtype=np.float16)

        #run ants
        best_path = []
        best_path_length = 0
        path = np.zeros((self.coef["n_vertices"]), dtype=int)

        for start_pos in range(0,self.coef["n_vertices"],self.coef["ants_step"]):
            path[0] = start_pos

            for i in range(self.coef["n_vertices"]-1):
                eliminated_weights = np.zeros((self.coef["n_vertices"]), dtype=np.float16)
                for j in range(i+1):
                    j = path[j]
                    eliminated_weights[j] = weights_memory[path[i]][j]
                weights = weights_memory[path[i]]-eliminated_weights
                #if force_debug: print(path)
                #if force_debug: print(np.round(weights, 4))
                selected_path = np.random.choice(len(weights), size = 1, p=np.divide(weights, np.sum(weights, dtype=np.float64), dtype = np.float32))[0]
                path[i+1] = selected_path
            path_length = self.func(path)
            for i in range(self.coef["n_vertices"]-1):
                delta_pheromones[path[i]][path[i+1]] += (self.coef["Q"] / path_length)
                delta_pheromones[path[i+1]][path[i]] += (self.coef["Q"] / path_length)
            delta_pheromones[path[0]][path[-1]] += (self.coef["Q"] / path_length)
            delta_pheromones[path[-1]][path[0]] += (self.coef["Q"] / path_length)
            if start_pos == 0:
                best_path = path
                best_path_length = path_length
            else:
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length
            if force_debug: print(path)
            if force_debug: print()
        if force_debug: print("---------")
        self.pheromones *= (1-self.coef["evaporation"])
        self.pheromones += cp.asarray(delta_pheromones)

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