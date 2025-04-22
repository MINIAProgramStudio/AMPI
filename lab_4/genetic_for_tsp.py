import numpy as np
from random import random
from tqdm import tqdm

"""
coef

"n_vertices": ,
"pop_size": ,
"children": ,
"m_switch_prob": ,
"m_pop_prob": ,
"""

class GTSP:
    def __init__(self, coef, path_eval_func, seeking_min = True):
        self.func = path_eval_func
        self.seeking_min = seeking_min
        self.coef = coef

        self.population = [np.arange(0,self.coef["n_vertices"]) for _ in range(self.coef["pop_size"])]
        for i in range(self.coef["pop_size"]):
            np.random.shuffle(self.population[i])
            self.population[i] = self.population[i].tolist()
        print(self.population)

    def sort_and_truncate(self):
        self.population.sort(key = self.func, reverse = not self.seeking_min)
        self.population = self.population[:self.coef["pop_size"]]

    def mutate_switch(self):
        for i in range(self.coef["pop_size"]):
            if random() < self.coef["m_switch_prob"]:
                a, b = int(random()*self.coef["pop_size"]), int(random()*self.coef["pop_size"])
                self.population[i][a], self.population[i][b] = self.population[i][b], self.population[i][a]

    def mutate_pop_insert(self):
        for i in range(self.coef["pop_size"]):
            if random() < self.coef["m_pop_prob"]:
                a, b = int(random() * self.coef["n_vertices"]), int(random() * self.coef["n_vertices"])
                if b>a: b-=1
                vertice = self.population[i].pop(a)
                self.population[i].insert(b, vertice)

    def generate_children(self):
        for i in range(self.coef["children"]):
            a, b = int(random() * self.coef["pop_size"]), int(random() * self.coef["pop_size"])
            a_parent = self.population[a]
            b_parent = self.population[b]
            connections = dict()
            reverse_connections = dict()
            for j in range(self.coef["n_vertices"]):
                vert_from = a_parent[j]
                vert_to_a = a_parent[(j+1)%self.coef["n_vertices"]]
                vert_to_b = b_parent[(b_parent.index(vert_from)+1)%self.coef["n_vertices"]]
                if vert_to_a == vert_to_b:
                    connections[vert_from] = vert_to_a
                    reverse_connections[vert_to_a] = vert_from

            path = [a_parent[0]] + [None]*(self.coef["n_vertices"]-1)

            start_index = 1
            while path[0] in reverse_connections.keys() and path[-1] is None:
                path.insert(0, reverse_connections[path[0]])
                path = path[:self.coef["n_vertices"]]
                start_index += 1
            for j in range(start_index,self.coef["n_vertices"]):
                if path[j-1] in connections.keys():
                    path[j] = connections[path[j-1]]
                else:
                    if j%2:
                        path[j] = a_parent[(a_parent.index(path[j-1])+1)%self.coef["n_vertices"]]
                    else:
                        path[j] = b_parent[(b_parent.index(path[j - 1]) + 1)%self.coef["n_vertices"]]
            self.population.append(path)

    def iter(self):
        #self.mutate_switch()
        self.mutate_pop_insert()
        self.generate_children()
        self.sort_and_truncate()

    def solve(self, iterations = 100, progressbar = False):
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc = "GTSP")
        for _ in iterator:
            self.iter()
        return(self.population[-1], self.func(self.population[-1]))