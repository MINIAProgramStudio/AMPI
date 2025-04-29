import numpy as np
from random import random
from tqdm import tqdm
import copy
import time

def de_loopilise(path, a_parent, b_parent):
    vert_from = path[-1]
    seek_a = a_parent[(a_parent.index(vert_from)+1)%len(a_parent)]
    seek_b = b_parent[(a_parent.index(vert_from)+1)%len(a_parent)]
    while seek_a in path and seek_b in path:
        seek_a = a_parent[(a_parent.index(seek_a) + 1) % len(a_parent)]
        seek_b = b_parent[(b_parent.index(seek_b) + 1) % len(a_parent)]
    if seek_a in path:
        path.append(seek_b)
    else:
        path.append(seek_a)
    return path


"""
coef

"n_vertices": ,
"pop_size": ,
"elitism": ,
"children": ,
"m_switch_prob": ,
"m_pop_prob": ,
"""

class GTSP:
    def __init__(self, coef, path_eval_func, seeking_min = True):
        self.startup_time = time.time()
        self.func = path_eval_func
        self.seeking_min = seeking_min
        self.coef = coef
        self.population = []
        if self.coef["elitism"]>self.coef["pop_size"]:
            self.coef["elitism"] = self.coef["pop_size"]
        if (not isinstance(self.coef["greed"],bool)) and seeking_min:

            for start in range(min(self.coef["pop_size"], self.coef["n_vertices"])):
                path = [start]
                for _ in range(self.coef["n_vertices"]-1):
                    distances = self.coef["greed"][path[-1]].tolist()
                    while True:
                        best_distance = distances.index(min(distances))
                        if best_distance in path:
                            distances[best_distance] = float("inf")
                        else:
                            break
                    path.append(best_distance)
                self.population.append(path)

        self.population += [np.random.permutation(self.coef["n_vertices"]).tolist() for _ in range(len(self.population),self.coef["pop_size"])]
        self.startup_time = time.time()-self.startup_time

    def reset(self):
        self.population = [np.arange(0, self.coef["n_vertices"]) for _ in range(self.coef["pop_size"])]
        for i in range(self.coef["pop_size"]):
            np.random.shuffle(self.population[i])
            self.population[i] = self.population[i].tolist()
    def sort_and_truncate(self):
        self.population.sort(key = self.func, reverse = not self.seeking_min)
        self.population = self.population[:self.coef["pop_size"]]

    def mutate_switch(self):
        for i in range(self.coef["elitism"]):
            if random() < self.coef["m_switch_prob"]:
                a, b = int(random()*self.coef["n_vertices"]), int(random()*self.coef["n_vertices"])
                self.population.append(copy.deepcopy(self.population[i]))
                self.population[-1][a], self.population[-1][b] = self.population[-1][b], self.population[-1][a]
        for i in range(self.coef["elitism"],self.coef["pop_size"]):
            if random() < self.coef["m_switch_prob"]:
                a, b = int(random()*self.coef["n_vertices"]), int(random()*self.coef["n_vertices"])
                self.population[i][a], self.population[i][b] = self.population[i][b], self.population[i][a]

    def mutate_pop_insert(self):
        for i in range(self.coef["elitism"]):
            if random() < self.coef["m_pop_prob"]:
                a, b = int(random() * self.coef["n_vertices"]), int(random() * self.coef["n_vertices"])
                if b>a: b-=1
                self.population.append(copy.deepcopy(self.population[i]))
                vertice = self.population[-1].pop(a)
                self.population[-1].insert(b, vertice)
        for i in range(self.coef["elitism"],self.coef["pop_size"]):
            if random() < self.coef["m_pop_prob"]:
                a, b = int(random() * self.coef["n_vertices"]), int(random() * self.coef["n_vertices"])
                if b>a: b-=1
                vertice = self.population[i].pop(a)
                self.population[i].insert(b, vertice)

    def generate_children(self):
        for i in range(self.coef["children"]):
            a, b = int(random() * self.coef["pop_size"]), int(random() * self.coef["pop_size"])
            if a == b:
                a, b = int(random() * self.coef["pop_size"]), int(random() * self.coef["pop_size"])
            a_parent = self.population[a]
            b_parent = self.population[b]
            path = [0]
            while len(path) < self.coef["n_vertices"]:
                vert_to_a = a_parent[(a_parent.index(path[-1])+1)%self.coef["n_vertices"]]
                vert_to_b = b_parent[(b_parent.index(path[-1])+1)%self.coef["n_vertices"]]
                if vert_to_a == vert_to_b or path[-1]%2:
                    if vert_to_a in path:
                        if vert_to_b in path:
                            path = de_loopilise(path, a_parent, b_parent)
                        else:
                            path.append(vert_to_b)
                    else:
                        path.append(vert_to_a)
                else:
                    if vert_to_b in path:
                        if vert_to_a in path:
                            path = de_loopilise(path, a_parent, b_parent)
                        else:
                            path.append(vert_to_a)
                    else:
                        path.append(vert_to_b)
            self.population.append(path)

    def iter(self):
        self.mutate_switch()
        self.mutate_pop_insert()
        self.generate_children()
        self.sort_and_truncate()

    def solve(self, iterations = 100, progressbar = False):
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc = "GTSP")
        for _ in iterator:
            self.iter()
        return(self.func(self.population[0]),self.population[0])

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="GTSP")
        self.sort_and_truncate()
        for _ in iterator:
            output.append(self.func(self.population[0]))
            self.iter()
        output.append(self.func(self.population[0]))
        return (self.func(self.population[0]), self.population[0], output)

    def anisolve(self, TSP, iterations = 100, step = 5, minimum_frame_time = 0.5):
        for i in range(iterations//step):
            t = time.time()
            for j in range(step):
                self.iter()

            TSP.draw_graph(path = self.population[0])
            time.sleep(max(t + minimum_frame_time - time.time(), 0))
        return (self.func(self.population[0]), self.population[0])

    def solve_seconds(self, seconds = 10):
        output = []
        start = time.time()-self.startup_time
        self.sort_and_truncate()
        while start + seconds > time.time():
            output.append([time.time()-start, self.func(self.population[0])])
            self.iter()
        output.append([time.time()-start, self.func(self.population[0])])
        return (self.func(self.population[0]), self.population[0], output)