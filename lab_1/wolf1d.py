from email.parser import Parser
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

class Wolf1D:
    def __init__(self, x, speed):
        self.x = x
        self.speed = speed

class Pack1D:
    def __init__(self, x_min, x_max, func, n_of_wolfs, speed_func = None, seeking_min = False):
        self.x_min = x_min
        self.x_max = x_max
        self.func = func
        if speed_func is None:
            def speed_func(rand):
                return rand*(x_max-x_min)/(n_of_wolfs)
        self.speed_func = speed_func
        self.seeking_min = seeking_min
        if n_of_wolfs >= 2:
            self.wolfs = [Wolf1D(x_min, self.speed_func(random())), # Вовк на мінімальному значенні
                          Wolf1D(x_max, self.speed_func(random())) # Вовк на максимальному значенні
                          ]
            if n_of_wolfs >=3:
                for i in range(2,n_of_wolfs):
                    self.wolfs.append(Wolf1D(random()*(x_max-x_min)+x_min, self.speed_func(random()))) # Вовки на випадкових позиціях
        else:
            raise Exception("too few wolfs")

    def find_prime(self):
        prime = 0
        if self.seeking_min:
            prime_value = float("inf")
            for i in range(len(self.wolfs)):
                value = self.func(self.wolfs[i].x)
                if value < prime_value:
                    prime = i
                    prime_value = value
        else:
            prime_value = -float("inf")
            for i in range(len(self.wolfs)):
                value = self.func(self.wolfs[i].x)
                if value > prime_value:
                    prime = i
                    prime_value = value
        return [prime, prime_value]

    def move(self, mult = 1):
        prime = self.find_prime()[0]
        for i in range(len(self.wolfs)):
            if i != prime:
                if self.wolfs[i].x < self.wolfs[prime].x:
                    self.wolfs[i].x += self.wolfs[i].speed*mult
                else:
                    self.wolfs[i].x -= self.wolfs[i].speed*mult
                self.wolfs[i].x = max(self.x_min, min(self.x_max, self.wolfs[i].x))

    def solve(self, iterations = 100, animate = False):
        if animate:
            fig, ax = plt.subplots()
            dots_x = [wolf.x for wolf in self.wolfs]
            dots_y = [self.func(wolf.x) for wolf in self.wolfs]
            prime_x = [self.wolfs[self.find_prime()[0]].x]
            prime_y = [self.find_prime()[1]]

            func_x = np.linspace(self.x_min, self.x_max, 1000)
            func_y = [self.func(i) for i in func_x]

            dots = ax.scatter(dots_x, dots_y, c="#0055ff", zorder = 5)
            prime = ax.scatter(prime_x, prime_y, s=75, c = "#ffaa00", zorder = 10)
            func = ax.plot(func_x, func_y, zorder = 0)

            fig.suptitle("Wolfs "+str(0) + "/" + str(iterations) + " Best: " + str(round(self.find_prime()[1],3)))
            def update(frame):
                self.move(10/iterations)
                dots_x = [wolf.x for wolf in self.wolfs]
                dots_y = [self.func(wolf.x) for wolf in self.wolfs]
                prime_x = [self.wolfs[self.find_prime()[0]].x]
                prime_y = [self.find_prime()[1]]

                dots.set_offsets(np.c_[dots_x, dots_y])
                prime.set_offsets(np.c_[[prime_x], [prime_y]])
                fig.suptitle("Wolfs "+str(frame+1)+"/"+str(iterations)+" Best: "+str(round(self.find_prime()[1],3)))
                if frame >=iterations-1:
                    ani.pause()
                return dots, prime

            writervideo = animation.PillowWriter(fps=2, bitrate=1800)
            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval = 500)
            # ani.save("gifs/wolf_latest.gif", writer = writervideo)
            plt.show()

        else:
            for i in range(iterations):
                self.move()
            return [self.wolfs[self.find_prime()[0]].x, self.find_prime()[1]]
