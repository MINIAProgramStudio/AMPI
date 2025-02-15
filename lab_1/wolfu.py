from email.parser import Parser
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import numpy as np
import math
from matplotlib import cm

def sign(a,b):
    if a>b:
        return 1
    else:
        return -1

class WolfU:
    def __init__(self, pos, speed):
        self.pos = pos
        self.speed = speed

class PackU:
    def __init__(self, minmax, func, n_of_wolfs, speed_func = None, seeking_min = False):
        self.minmax = minmax
        if speed_func is None:
            def speed_func(rand):
                return rand*math.sqrt(sum([(dim[1]-dim[0])**2 for dim in minmax]))/(n_of_wolfs)
        self.speed_func = speed_func
        self.seeking_min = seeking_min
        self.wolfs = []
        self.func = func
        for i in range(n_of_wolfs):
            self.wolfs.append(WolfU([
                random()*(dim[1]-dim[0]) + dim[0] for dim in minmax
            ], self.speed_func(random())))

    def find_prime(self):
        prime = 0
        if self.seeking_min:
            prime_value = float("inf")
            for i in range(len(self.wolfs)):
                value = self.func(self.wolfs[i].pos)
                if value < prime_value:
                    prime = i
                    prime_value = value
        else:
            prime_value = -float("inf")
            for i in range(len(self.wolfs)):
                value = self.func(self.wolfs[i].pos)
                if value > prime_value:
                    prime = i
                    prime_value = value
        return [prime, prime_value]

    def move(self, mult = 1):
        prime = self.find_prime()[0]
        for i in range(len(self.wolfs)):
            if i != prime:
                distance = math.sqrt(sum([(self.wolfs[prime].pos[dim] - self.wolfs[i].pos[dim])**2 for dim in range(len(self.minmax))]))
                prime_projection = [(self.wolfs[prime].pos[dim]-self.wolfs[i].pos[dim])/distance for dim in range(len(self.minmax))]
                self.wolfs[i].pos = [self.wolfs[i].pos[dim] + self.wolfs[i].speed * mult * prime_projection[dim] for dim in range(len(self.minmax))]

    def solve(self, iterations = 100, animate = False, target = 0):
        if animate:
            px = list(range(iterations+1))
            py = [abs(self.find_prime()[1] - target)]
            fig, ax = plt.subplots()
            for i in range(iterations):
                self.move()
                py.append(abs(self.find_prime()[1] - target))

            ax.set_yscale("log")


            graph = ax.plot(px,py)[0]
            plt.show()
            return self.find_prime()[1]
        else:
            for i in range(iterations):
                self.move(10/iterations)
            return self.find_prime()[1]
