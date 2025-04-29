#import numpy as np
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
        self.check_path = TSP.check_path
        self.antilengths = cp.power(cp.divide(1,cp.asarray(TSP.matrix, dtype=cp.float16)),self.coef["b"])
        self.coef["n_vertices"] = len(TSP.matrix)
        if not "ants_step" in self.coef.keys():
            self.coef["ants_step"] = 1
        
        self.pheromones = cp.ones((self.coef["n_vertices"], self.coef["n_vertices"]),dtype=cp.float16)

        self.startup_time = time.time() - self.startup_time
        print(self.startup_time)

    def func(self,path):
        return self.check_path(path.tolist())

    def reset(self):
        self.pheromones = cp.ones((self.coef["n_vertices"], self.coef["n_vertices"]))

    def iter(self, force_debug = False):
        time_memory = 0

        weights_memory = cp.multiply(cp.power(self.pheromones, self.coef["a"]),self.antilengths)
        delta_pheromones = cp.zeros((self.coef["n_vertices"], self.coef["n_vertices"]),dtype=cp.float16)

        #run ants

        best_path = []
        best_path_length = 0
        paths = cp.zeros((self.coef["n_vertices"],self.coef["n_vertices"]), dtype=int)

        for start_pos in range(0,self.coef["n_vertices"],self.coef["ants_step"]):

            paths[start_pos][0] = start_pos
            elimination_vector = cp.ones((self.coef["n_vertices"]))

            for i in range(self.coef["n_vertices"]-1):

                weights = cp.multiply(weights_memory[paths[start_pos][i]],elimination_vector)

                #if force_debug: print(path)
                #if force_debug: print(cp.round(weights, 4))
                time_memory -= time.time()
                selected_path = cp.random.choice(self.coef["n_vertices"], size = 1, p=cp.divide(weights, cp.sum(weights, dtype=cp.float64), dtype = cp.float32))[0]
                time_memory += time.time()
                paths[start_pos][i+1] = selected_path
                elimination_vector[i+1] = 0

        paths_lengths = list(map(self.func, paths))


        for k in range(self.coef["n_vertices"]):
            for i in range(self.coef["n_vertices"]-1):
                delta_pheromones[paths[k][i]][paths[k][i+1]] += (self.coef["Q"] / paths_lengths[k])
                delta_pheromones[paths[k][i+1]][paths[k][i]] += (self.coef["Q"] / paths_lengths[k])
            delta_pheromones[paths[k][0]][paths[k][-1]] += (self.coef["Q"] / paths_lengths[k])
            delta_pheromones[paths[k][-1]][paths[k][0]] += (self.coef["Q"] / paths_lengths[k])
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
        self.pheromones += cp.asarray(delta_pheromones)
        print(time_memory)
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
                if force_debug: print(cp.round(self.pheromones,3).astype(float))
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