from random import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation, cm
from time import time

"""
coef list
"b_max": ,
"gamma": ,
"alpha": ,
"pop_size": ,
"pos_min": ,
"pos_max": ,
"dim": ,
"""

class FFSolver:
    def __init__(self, coef, func, seeking_min = False):
        self.c = coef
        self.func = func
        self.seeking_min = seeking_min
        self.pop = [np.array([random()*(self.c["pos_max"][d] - self.c["pos_min"][d]) + self.c["pos_min"][d] for d in range(self.c["dim"])]) for _ in range(self.c["pop_size"])]

    def reset(self):
        self.pop = [np.array([random() * (self.c["pos_max"][d] - self.c["pos_min"][d]) + self.c["pos_min"][d] for d in
                              range(self.c["dim"])]) for _ in range(self.c["pop_size"])]

    def iter(self):
        self.pop.sort(key = self.func, reverse = not self.seeking_min)
        for k in range(1, self.c["pop_size"]):
            for l in range(k):
                b = self.c["b_max"]*np.exp(-self.c["gamma"]*sum([
                    (self.pop[k][d]-self.pop[l][d])**2 for d in range(self.c["dim"])
                ]))
                self.pop[k] += b*(self.pop[l]-self.pop[k]) + self.c["alpha"]*(random() - 0.5)
                self.pop[k] = np.minimum(self.c["pos_max"], np.maximum(self.c["pos_min"], self.pop[k]))

    def solve(self, itertaions = 100, progressbar = False):
        iterator = range(itertaions)
        if progressbar:
            iterator = tqdm(iterator, desc = "FF")
        for _ in iterator:
            self.iter()
        self.pop.sort(key=self.func, reverse=not self.seeking_min)
        return (self.func(self.pop[0]), self.pop[0])

    def anisolve(self, iterations = 100, save = False):
        if self.c["dim"] == 2:
            fig = plt.figure()
            ax = plt.axes(projection="3d", computed_zorder=False)
            dots_x = [pop[0] for pop in self.pop]
            dots_y = [pop[1] for pop in self.pop]
            dots_z = [self.func(pop) for pop in self.pop]
            prime_x = [self.pop[0][0]]
            prime_y = [self.pop[0][1]]
            prime_z = [self.func(self.pop[0])]

            func_x = np.linspace(self.c["pos_min"][0], self.c["pos_max"][0], 100)
            func_y = np.linspace(self.c["pos_min"][1], self.c["pos_max"][1], 100)
            FUNC_X, FUNC_Y = np.meshgrid(func_x, func_y)

            FUNC_Z = self.func([FUNC_X, FUNC_Y])

            surface = ax.plot_surface(FUNC_X, FUNC_Y, FUNC_Z, cmap=cm.coolwarm,
                                      linewidth=0, antialiased=False, zorder=0)
            dots = ax.scatter(dots_x, dots_y, dots_z, c="#ff0000", zorder=5)
            prime = ax.scatter(prime_x, prime_y, prime_z, s=75, c="#ffff00", zorder=10)

            fig.suptitle("FF " + str(0) + "/" + str(iterations) + " Best: " + str(round(self.func(self.pop[0]), 3)))

            def update(frame):
                self.iter()
                dots_x = [pop[0] for pop in self.pop]
                dots_y = [pop[1] for pop in self.pop]
                dots_z = [self.func(pop) for pop in self.pop]
                prime_x = [self.pop[0][0]]
                prime_y = [self.pop[0][1]]
                prime_z = [self.func(self.pop[0])]

                dots.set_offsets(np.c_[dots_x, dots_y])
                dots.set_3d_properties(dots_z, zdir='z')
                prime.set_offsets(np.c_[prime_x, prime_y])
                prime.set_3d_properties(prime_z, zdir='z')
                fig.suptitle(
                    "FF " + str(frame + 1) + "/" + str(iterations) + " Best: " + str(round(self.func(self.pop[0]))))
                if frame >= iterations - 1:
                    ani.pause()
                return dots, prime

            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=50)
            if save:
                writervideo = animation.PillowWriter(fps=2, bitrate=1800)
                ani.save("gifs/ff_latest.gif", writer = writervideo)
            plt.show()
            return (self.pop[0], self.func(self.pop[0]))
        else:
            px = list(range(iterations + 1))
            py = [abs(self.func(self.pop[0]))]
            fig, ax = plt.subplots()
            for i in range(iterations):
                self.iter()
                py.append(abs(self.func(self.pop[0])))

            ax.set_yscale("log")

            graph = ax.plot(px, py)[0]
            if save:
                plt.savefig("gifs/pso_lates.png")
            plt.show()
            return (self.func(self.pop[0]), self.pop[0])

    def solve_stats(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="FF")
        self.pop.sort(key=self.func, reverse=not self.seeking_min)
        for _ in iterator:
            output.append(self.func(self.pop[0]))
            self.iter()
        output.append(self.func(self.pop[0]))
        return (self.func(self.pop[0]), self.pop[0], output)

    def solve_time(self, iterations = 100, progressbar = False):
        output = []
        iterator = range(iterations)
        if progressbar:
            iterator = tqdm(iterator, desc="FF")
        self.pop.sort(key=self.func, reverse=not self.seeking_min)
        start = time()
        for _ in iterator:
            output.append(time()-start)
            self.iter()
        output.append(time()-start)
        return (self.func(self.pop[0]), self.pop[0], output)
