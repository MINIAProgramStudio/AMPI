from random import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation



class Genetic2D:
    def __init__(self, x_min, x_max, y_min, y_max, func, pop_size, children_count, bit_length = 8, seeking_min = False, mutation_prob = 0.1):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.pop_size = pop_size
        self.children_count = children_count
        self.bit_length = bit_length
        self.seeking_min = seeking_min
        self.mutation_prob = mutation_prob
        self.func = func
        self.memory = {}
        self.population = [0]*(self.pop_size + self.children_count)
        for i in range(pop_size+children_count):
            self.population[i] = int(random()*(2**self.bit_length))
        self.selection()

    def mutate_dna(self, dna):
        return dna ^ 2 ** int(random() * self.bit_length)

    def evaluate_dna(self, dna):
        if dna not in self.memory.keys():
            self.memory[dna] = self.func(*self.dna_pos(dna))
        return self.memory[dna]

    def dna_pos(self, dna):
        return [(dna%(2 ** (self.bit_length//2))) * (self.x_max - self.x_min) / (2 ** (self.bit_length//2)) + self.x_min,
        (dna//(2 ** (self.bit_length//2))) * (self.y_max - self.y_min) / (2 ** (self.bit_length//2)) + self.y_min]

    def selection(self):
        self.population.sort(reverse= not self.seeking_min, key=self.evaluate_dna)

    def iteration(self):
        # create children
        separator = int(self.bit_length // 2)
        for i in range(int(self.children_count/2)):
            dna_2 = self.population[int(random()*self.pop_size)]
            dna_1 = self.population[int(random()*self.pop_size)]
            child_1 = dna_1 & (2**separator-1) + dna_2 & (2**self.bit_length-2**separator)
            child_2 = dna_2 & (2**separator-1) + dna_1 & (2**self.bit_length-2**separator)
            self.population[self.pop_size + i] = child_1
            self.population[self.pop_size + i + 1] = child_2
        for i in range(1,len(self.population)):
            if random()<self.mutation_prob:
                self.population[i] = self.mutate_dna(self.population[i])
        self.selection()

    def solve(self, iterations = 100, animate = False):
        if animate:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis")

            func_x = np.linspace(self.x_min, self.x_max, 25)
            func_y = np.linspace(self.y_min, self.y_max, 25)
            FUNC_X, FUNC_Y = np.meshgrid(func_x, func_y)
            FUNC_Z = self.func(FUNC_X, FUNC_Y)

            ax.plot_wireframe(FUNC_X, FUNC_Y, FUNC_Z, color="black", zorder=0)
            dots = ax.scatter([], [], [], c="#ff0000", zorder=5, label="Population")
            prime = ax.scatter([], [], [], s=75, c="#ffff00", zorder=10, label="Best Individual")

            ax.legend()
            ax.grid(True)

            def update(frame):
                self.iteration()
                fig.suptitle("Genetic" + str(frame + 1) + "/" + str(iterations) + " Best: " + str(round(self.evaluate_dna(self.population[0]), 3)))
                x_coords = [self.dna_pos(p)[0] for p in self.population]
                y_coords = [self.dna_pos(p)[1] for p in self.population]
                z_coords = [self.evaluate_dna(p) for p in self.population]

                dots._offsets3d = (x_coords, y_coords, z_coords)
                prime._offsets3d = ([self.dna_pos(self.population[0])[0]],
                                    [self.dna_pos(self.population[0])[1]],
                                    [self.evaluate_dna(self.population[0])])

                return dots, prime

            # writervideo = animation.PillowWriter(fps=2, bitrate=1800)
            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=500)
            # ani.save("gifs/wolf_latest.gif", writer = writervideo)
            plt.show()
        else:
            for i in range(iterations):
                self.iteration()
        return self.evaluate_dna(self.population[0])