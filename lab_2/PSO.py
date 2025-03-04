import numpy as np
from tqdm import tqdm

class PSOParticle:
    def __init__(self, coef, func, seeking_min):
        self.c = coef
        self.pos = np.random.random_sample(self.c["dim"]) * (self.c["pos_max"]-self.c["pos_min"]) + self.c["pos_min"]
        self.speed = np.random.random_sample(self.c["dim"]) * (self.c["speed_max"]-self.c["speed_min"])/10 + self.c["speed_min"]
        self.best_pos = self.pos
        self.func = func
        self.seeking_min = seeking_min
        self.best_val = self.func(self.pos)

    def update(self, global_best_pos, r1, r2):
        # оновлення швидкості
        self.speed += self.c["a1"] * (self.best_pos - self.pos) *  + self.c["a2"] * (global_best_pos - self.pos) * np.random.random_sample(self.c["dim"])
        self.speed = np.minimum(self.c["speed_max"], np.maximum(self.c["speed_min"], self.speed))
        # оновлення позиції
        self.pos += self.speed
        for d in range(self.c["dim"]):
            if self.pos[d] < self.c["pos_min"][d]:
                self.pos[d] = self.c["pos_min"][d] + abs(self.pos[d]-self.c["pos_min"][d])
            if self.pos[d] > self.c["pos_max"][d]:
                self.pos[d] = self.c["pos_max"][d] - abs(self.pos[d] - self.c["pos_max"][d])
        # оновлення найкращої позиції
        result = self.func(self.pos)
        if (result > self.best_val) ^ self.seeking_min :
            self.best_pos = self.pos
            self.best_val = result
            return (self.pos, result)
        else:
            return None

"""
coef list:
"a1": ,#acceleration number
"a2": ,#acceleration number
"max_iter": ,#max iterations
"pop_size": ,#population size
"dim": ,#dimensions
"pos_min": ,#vector of minimum positions
"pos_max": ,#vector of maximum positions
"speed_min": ,#vector of min speed
"speed_max": ,#vector of max speed
"""


class PSOSolver:
    def __init__(self, coef, func, seeking_min = False):
        self.c = coef
        self.func = func
        self.seeking_min = seeking_min
        self.pop = [PSOParticle(self.c, self.func, self.seeking_min) for i in range(self.c["pop_size"])]
        personal_best = [particle.best_val for particle in self.pop]
        if self.seeking_min:
            best_i = np.argmin(personal_best)
        else:
            best_i = np.argmax(personal_best)
        print(best_i)
        self.best_pos = self.pop[best_i].best_pos
        self.best_val = self.pop[best_i].best_val

    def iter(self):
        new_best_val = self.best_val
        new_best_pos = self.best_pos
        r1 = np.random.random_sample(self.c["dim"])
        r2 = np.random.random_sample(self.c["dim"])
        for particle in self.pop:
            result = particle.update(self.best_pos, r1, r2)
            if result:
                if (result[1] > self.best_val) ^ self.seeking_min:
                    new_best_val = result[1]
                    new_best_pos = result[0]
        self.best_val = new_best_val
        self.best_pos = new_best_pos

    def solve(self, animate = False):
        if animate:
            pass
        else:
            for iter in tqdm(range(self.c["max_iter"]), desc="PSO"):
                self.iter()
            return (self.best_val, self.best_pos)