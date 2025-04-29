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
"""

class AntSolver:
    def __init__(self, coef, TSP):
        self.startup_time = time.time()
        self.coef = coef
        self.func = TSP.check_path

        self.lengths = TSP.matrix
        self.antilengths = np.power(1/TSP.matrix,self.coef["b"])
        self.coef["n_vertices"] = len(TSP.matrix)
        
        self.pheromones = np.ones((self.coef["n_vertices"], self.coef["n_vertices"]))
        
        self.startup_time = time.time() - self.startup_time

    def reset(self):
        self.pheromones = np.ones((self.coef["n_vertices"], self.coef["n_vertices"]))

    def iter(self, force_debug = False):
        weights_memory = np.multiply(np.power(self.pheromones, self.coef["a"]),self.antilengths)
        # generate weights
        for start_pos in range(self.coef["n_vertices"]):
            for end_pos in range(start_pos):
                weights_memory[start_pos][end_pos] = weights_memory[end_pos][start_pos]

            weights_memory[start_pos][start_pos] = 0
        #balance weights
        for start_pos in range(self.coef["n_vertices"]):
            weights_memory[start_pos] /= np.sum(weights_memory[start_pos])
        delta_pheromones = np.zeros((self.coef["n_vertices"], self.coef["n_vertices"]))
        #run ants
        best_path = []
        best_path_length = 0
        path = np.zeros((self.coef["n_vertices"]), dtype=int)
        eliminated_weights = np.zeros((self.coef["n_vertices"]))
        for start_pos in range(self.coef["n_vertices"]):
            path[0] = start_pos

            for i in range(self.coef["n_vertices"]-1):

                for j in range(self.coef["n_vertices"]):
                    if j in path[:i+1]:
                        eliminated_weights[j] = weights_memory[path[i]][j]
                    else:
                        eliminated_weights[j] = 0
                if all(eliminated_weights != 0):
                    break
                weights = (weights_memory[path[i]]-eliminated_weights) / (np.sum(weights_memory[path[i]]-eliminated_weights))
                if force_debug: print(path)
                if force_debug: print(np.round(weights, 4))


                selected_path = np.random.choice(len(weights), p=weights)
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
        self.pheromones += delta_pheromones
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
            if i == 0:
                best_path = result[1]
                best_length = result[0]
            else:
                if best_length > result[0]:
                    best_path = result[1]
                    best_length = result[0]
            output.append(self.func(best_path))
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
            output.append([time.time()-start,self.func(best_path)])
        return (best_length, best_path, output)