from random import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation, cm
from time import time
"""
coef list
"delta": ,
"n_max": ,
"alpha": ,
"pop_size": ,
"l_s": ,
"l_es": ,
"z_e": ,
"z_o": ,
"dim": ,
"pos_min": ,
"pos_max": 
"""
class BEEHive:
    def __init__(self, coef, func, seeking_min = False):
        self.c = coef
        self.func = func
        self.seeking_min = seeking_min
        self.best_pos = [random()*(self.c["pos_max"][d]-self.c["pos_min"][d])+self.c["pos_min"][d] for d in range(self.c["dim"])]
        self.best_value = self.func(self.best_pos)
        self.bees = [[random()*(self.c["pos_max"][d]-self.c["pos_min"][d])+self.c["pos_min"][d] for d in range(self.c["dim"])]
                     for _ in range(self.c["pop_size"])]
        self.iterations = 0

    def reset(self):
        self.best_pos = [random() * (self.c["pos_max"][d] - self.c["pos_min"][d]) + self.c["pos_min"][d] for d in
                         range(self.c["dim"])]
        self.best_value = self.func(self.best_pos)
        self.bees = [[random() * (self.c["pos_max"][d] - self.c["pos_min"][d]) + self.c["pos_min"][d] for d in
                      range(self.c["dim"])]
                     for _ in range(self.c["pop_size"])]
        self.iterations = 0

    def iter(self):
        self.iterations += 1
        self.bees.sort(key = self.func, reverse = not self.seeking_min)
        pop_best = self.func(self.bees[0])
        if pop_best > self.best_value and not self.seeking_min:
            self.best_pos = self.bees[0]
            self.best_value = pop_best
        elif pop_best < self.best_value and not self.seeking_min:
            self.best_pos = self.bees[0]
            self.best_value = pop_best

        for l in range(self.c["l_s"]):
            if l <= self.c["l_es"]:
                z = self.c["z_e"]
            else:
                z = self.c["z_o"]
            nu = self.c["n_max"]*self.c["alpha"]**self.iterations

            x_z = [[
                np.maximum(self.c["pos_min"], np.minimum(self.c["pos_max"],
                self.bees[l][d] + nu * self.c["delta"]*(self.c["pos_max"][d] - self.c["pos_min"][d]) * (-1 + 2*random()))[0])[0]
                for d in range(self.c["dim"])
            ] for _ in range(z)]
            x_z.sort(key = self.func, reverse = not self.seeking_min)
            if self.func(x_z[0]) > self.func(self.bees[l]) and not self.seeking_min:
                self.bees[l] = x_z[0]
            elif self.func(x_z[0]) < self.func(self.bees[l]) and self.seeking_min:
                self.bees[l] = x_z[0]

        for l in range(self.c["l_s"], self.c["pop_size"]):
            self.bees[l] = [random()*(self.c["pos_max"][d]-self.c["pos_min"][d])+self.c["pos_min"][d] for d in range(self.c["dim"])]

    def solve(self, itertaions = 100, progressbar = False):
        iterator = range(itertaions)
        if progressbar:
            iterator = tqdm(iterator, desc = "BEE")
        for _ in iterator:
            self.iter()
        self.bees.sort(key=self.func, reverse=not self.seeking_min)
        return (self.func(self.bees[0]), self.bees[0])

    def anisolve(self, iterations = 100, save = False):
        if self.c["dim"] == 2:
            fig = plt.figure()
            ax = plt.axes(projection="3d", computed_zorder=False)
            dots_x = [bee[0] for bee in self.bees]
            dots_y = [bee[1] for bee in self.bees]
            dots_z = [self.func(bee) for bee in self.bees]
            prime_x = [self.bees[0][0]]
            prime_y = [self.bees[0][1]]
            prime_z = [self.func(self.bees[0])]

            func_x = np.linspace(self.c["pos_min"][0], self.c["pos_max"][0], 100)
            func_y = np.linspace(self.c["pos_min"][1], self.c["pos_max"][1], 100)
            FUNC_X, FUNC_Y = np.meshgrid(func_x, func_y)

            FUNC_Z = self.func([FUNC_X, FUNC_Y])

            surface = ax.plot_surface(FUNC_X, FUNC_Y, FUNC_Z, cmap=cm.coolwarm,
                                      linewidth=0, antialiased=False, zorder=0)
            dots = ax.scatter(dots_x, dots_y, dots_z, c="#ff0000", zorder=5)
            prime = ax.scatter(prime_x, prime_y, prime_z, s=75, c="#ffff00", zorder=10)

            fig.suptitle("BEE " + str(0) + "/" + str(iterations) + " Best: " + str(round(self.func(self.bees[0]), 3)))

            def update(frame):
                self.iter()
                dots_x = [bee[0] for bee in self.bees]
                dots_y = [bee[1] for bee in self.bees]
                dots_z = [self.func(bee) for bee in self.bees]
                prime_x = [self.bees[0][0]]
                prime_y = [self.bees[0][1]]
                prime_z = [self.func(self.bees[0])]

                dots.set_offsets(np.c_[dots_x, dots_y])
                dots.set_3d_properties(dots_z, zdir='z')
                prime.set_offsets(np.c_[prime_x, prime_y])
                prime.set_3d_properties(prime_z, zdir='z')
                fig.suptitle(
                    "BEE " + str(frame + 1) + "/" + str(iterations) + " Best: " + str(round(self.func(self.bees[0]), 3)))
                if frame >= iterations - 1:
                    ani.pause()
                return dots, prime

            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=50)
            if save:
                writervideo = animation.PillowWriter(fps=2, bitrate=1800)
                ani.save("gifs/pso_latest.gif", writer = writervideo)
            plt.show()
            return (self.best_value, self.best_pos)
        else:
            px = list(range(iterations + 1))
            py = [abs(self.best_value)]
            fig, ax = plt.subplots()
            for i in range(iterations):
                self.iter()
                py.append(abs(self.best_value))

            ax.set_yscale("log")

            graph = ax.plot(px, py)[0]
            if save:
                plt.savefig("gifs/bee_lates.png")
            plt.show()
            return (self.best_value, self.best_pos)

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="BEE")
        self.bees.sort(key=self.func, reverse=not self.seeking_min)
        for _ in iterator:
            output.append(self.func(self.bees[0]))
            self.iter()
        output.append(self.func(self.bees[0]))
        return (self.func(self.bees[0]), self.bees[0], output)

    def solve_time(self, iterations = 100, progressbar = False):
        output = []

        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="BEE")

        self.bees.sort(key=self.func, reverse=not self.seeking_min)
        start = time()
        for _ in iterator:
            output.append(time()-start)
            self.iter()
        output.append(time()-start)
        return (self.func(self.bees[0]), self.bees[0], output)