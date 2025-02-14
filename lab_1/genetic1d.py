from random import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation



class Genetic1D:
    def __init__(self, x_min, x_max, func, pop_size, children_count, bit_length = 8, seeking_min = False, mutation_prob = 0.1):
        self.x_min = x_min
        self.x_max = x_max

        self.pop_size = pop_size
        self.children_count = children_count
        self.bit_length = bit_length
        self.seeking_min = seeking_min
        self.mutation_prob = mutation_prob
        self.func = func
        self.memory = {}
        self.population = [0]*(self.pop_size + self.children_count)
        for i in range(pop_size+children_count):
            self.population[i] = int(random()*2**bit_length)
        self.selection()

    def mutate_dna(self, dna):
        return dna ^ 2 ** int(random() * self.bit_length)

    def evaluate_dna(self, dna):
        if dna not in self.memory.keys():
            self.memory[dna] = self.func(self.dna_pos(dna))
        return self.memory[dna]

    def dna_pos(self, dna):
        return dna * (self.x_max - self.x_min) / (2 ** self.bit_length) + self.x_min

    def selection(self):
        self.population.sort(reverse= not self.seeking_min, key=self.evaluate_dna)

    def iteration(self):
        # create children
        old_pop = self.population
        separator = int(self.bit_length / 2)
        for i in range(int(self.children_count/2)):
            dna_2 = old_pop[int(random()*len(old_pop))]
            dna_1 = old_pop[int(random()*len(old_pop))]
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
            fig, ax = plt.subplots()
            dots_x = [self.dna_pos(pop) for pop in self.population]
            dots_y = [self.evaluate_dna(pop) for pop in self.population]
            prime_x = [self.dna_pos(self.population[0])]
            prime_y = [self.evaluate_dna(self.population[0])]

            func_x = np.linspace(self.x_min, self.x_max, 1000)
            func_y = [self.func(i) for i in func_x]

            dots = ax.scatter(dots_x, dots_y, c="#0055ff", zorder=5)
            prime = ax.scatter(prime_x, prime_y, s=75, c="#ffaa00", zorder=10)
            func = ax.plot(func_x, func_y, zorder=0)

            fig.suptitle("Genetic "+str(0) + "/" + str(iterations) + " Best: " + str(round(prime_y[0],3)))

            def update(frame):
                self.iteration()
                dots_x = [self.dna_pos(pop) for pop in self.population]
                dots_y = [self.evaluate_dna(pop) for pop in self.population]
                prime_x = [self.dna_pos(self.population[0])]
                prime_y = [self.evaluate_dna(self.population[0])]

                dots.set_xdata = dots_x
                dots.set_ydata = dots_y

                dots.set_offsets(np.c_[dots_x, dots_y])
                prime.set_offsets(np.c_[[prime_x], [prime_y]])
                fig.suptitle("Genetic "+str(frame+1) + "/" + str(iterations) + " Best: " + str(round(prime_y[0],3)))
                if frame >= iterations - 1:
                    ani.pause()
                return dots, prime

            writervideo = animation.PillowWriter(fps=2, bitrate=1800)
            # ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval=500)
            # ani.save("gifs/genetic_latest.gif", writer=writervideo)
            plt.show()
        else:
            for i in range(iterations):
                self.iteration()
        return self.evaluate_dna(self.population[0])