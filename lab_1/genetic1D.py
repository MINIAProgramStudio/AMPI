from random import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Pop1D:
    def __init__(self, bit_length, dna = None):
        if dna == None:
            self.dna = []
            for i in range(bit_length):
                self.dna.append(False)
        else:
            self.dna = dna
        for i in range(bit_length):
           if random() > 0.5:
               self.dna[i] = True

    def __str__(self):
        output = ""
        for i in self.dna:
            if i:
                output += "1"
            else:
                output += "0"
        return output

    def mutate(self):
        i = int(random()*len(self.dna))
        self.dna[i] = not self.dna[i]

class Genetic1D:
    def __init__(self, x_min, x_max, func, pop_size, children_count, bit_length = 8, seeking_min = False, mutation_prob = 0.1):
        self.x_min = x_min
        self.x_max = x_max
        self.func = func
        self.pop_size = pop_size
        self.children_count = children_count
        self.bit_length = bit_length
        self.seeking_min = seeking_min
        self.mutation_prob = mutation_prob

        self.population = []
        for i in range(pop_size):
            self.population.append(Pop1D(self.bit_length))

    def evaluate_dna(self, pop):
        x = int(str(pop),2)*(self.x_max-self.x_min)/(2**self.bit_length)+self.x_min
        return self.func(x)

    def dna_pos(self, pop):
        return int(str(pop), 2) * (self.x_max - self.x_min) / (2 ** self.bit_length) + self.x_min

    def selection(self):
        self.population.sort(reverse=not self.seeking_min, key=self.evaluate_dna)
        self.population = self.population[:self.pop_size]

    def iteration(self):
        # create children
        old_pop = self.population
        for i in range(int(self.children_count/2)):
            separator = int(random()*(self.bit_length))
            dna_1 = old_pop[int(random()*len(old_pop))]
            dna_2 = old_pop[int(random()*len(old_pop))]
            while dna_2 == dna_1:
                dna_2 = old_pop[int(random() * len(old_pop))]
            child_1 = Pop1D(self.bit_length, dna_1.dna[:separator] + dna_2.dna[separator:])
            child_2 = Pop1D(self.bit_length,dna_2.dna[:separator] + dna_1.dna[separator:])
            self.population.append(child_1)
            self.population.append(child_2)
        for i in range(1,len(self.population)):
            if random()<self.mutation_prob:
                self.population[i].mutate()
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
            for i in tqdm(range(iterations)):
                self.iteration()
        return self.evaluate_dna(self.population[0])