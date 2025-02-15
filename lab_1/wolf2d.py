from email.parser import Parser
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import numpy as np
import math
from matplotlib import cm

class Wolf2D:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

class Pack2D:
    def __init__(self, x_min, x_max, y_min, y_max, func, n_of_wolfs, speed_func = None, seeking_min = False):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.func = func
        if speed_func is None:
            def speed_func(rand):
                return rand*math.sqrt((x_max-x_min)**2+(y_max-y_min)**2)/(n_of_wolfs)
        self.speed_func = speed_func
        self.seeking_min = seeking_min
        if n_of_wolfs >= 4:
            self.wolfs = [Wolf2D(x_min, y_min, self.speed_func(random())),
                          Wolf2D(x_min, y_max, self.speed_func(random())),
                          Wolf2D(x_max, y_min, self.speed_func(random())),
                          Wolf2D(x_max, y_max, self.speed_func(random())),
                          ]
            if n_of_wolfs >=5:
                for i in range(2,n_of_wolfs):
                    self.wolfs.append(Wolf2D(random()*(x_max-x_min)+x_min, random()*(y_max-y_min)+y_min, self.speed_func(random()))) # Вовки на випадкових позиціях
        else:
            raise Exception("too few wolfs")

    def find_prime(self):
        prime = 0
        if self.seeking_min:
            prime_value = float("inf")
            for i in range(len(self.wolfs)):
                value = self.func(self.wolfs[i].x, self.wolfs[i].y)
                if value < prime_value:
                    prime = i
                    prime_value = value
        else:
            prime_value = -float("inf")
            for i in range(len(self.wolfs)):
                value = self.func(self.wolfs[i].x, self.wolfs[i].y)
                if value > prime_value:
                    prime = i
                    prime_value = value
        return [prime, prime_value]

    def move(self, mult = 1):
        prime = self.find_prime()[0]
        for i in range(len(self.wolfs)):
            if i != prime:
                distance = math.sqrt((self.wolfs[prime].x-self.wolfs[i].x)**2 + (self.wolfs[prime].y-self.wolfs[i].y)**2)
                prime_cos = (self.wolfs[prime].x-self.wolfs[i].x)/distance
                prime_sin = (self.wolfs[prime].y-self.wolfs[i].y)/distance
                self.wolfs[i].x += self.wolfs[i].speed*prime_cos
                self.wolfs[i].y += self.wolfs[i].speed * prime_sin

    def solve(self, iterations = 100, animate = False):
        if animate:
            fig = plt.figure()
            ax = plt.axes(projection = "3d",computed_zorder=False)
            dots_x = [wolf.x for wolf in self.wolfs]
            dots_y = [wolf.y for wolf in self.wolfs]
            dots_z = [self.func(wolf.x, wolf.y) for wolf in self.wolfs]
            prime_x = [self.wolfs[self.find_prime()[0]].x]
            prime_y = [self.wolfs[self.find_prime()[0]].y]
            prime_z = [self.find_prime()[1]]

            func_x = np.linspace(self.x_min, self.x_max, 100)
            func_y = np.linspace(self.y_min, self.y_max, 100)
            FUNC_X, FUNC_Y = np.meshgrid(func_x, func_y)

            FUNC_Z = self.func(FUNC_X, FUNC_Y)

            surface = ax.plot_surface(FUNC_X, FUNC_Y, FUNC_Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, zorder = 0)
            dots = ax.scatter(dots_x, dots_y, dots_z, c="#ff0000", zorder = 5)
            prime = ax.scatter(prime_x, prime_y, prime_z, s=75, c = "#ffff00", zorder = 10)


            fig.suptitle("Wolfs "+str(0) + "/" + str(iterations) + " Best: " + str(round(self.find_prime()[1],3)))
            def update(frame):
                self.move(10/iterations)
                dots_x = [wolf.x for wolf in self.wolfs]
                dots_y = [wolf.y for wolf in self.wolfs]
                dots_z = [self.func(wolf.x, wolf.y) for wolf in self.wolfs]
                prime_x = [self.wolfs[self.find_prime()[0]].x]
                prime_y = [self.wolfs[self.find_prime()[0]].y]
                prime_z = [self.find_prime()[1]]

                dots.set_offsets(np.c_[dots_x, dots_y])
                dots.set_3d_properties(dots_z, zdir='z')
                prime.set_offsets(np.c_[prime_x, prime_y])
                prime.set_3d_properties(prime_z, zdir='z')
                fig.suptitle("Wolfs "+str(frame+1)+"/"+str(iterations)+" Best: "+str(round(self.find_prime()[1],3)))
                if frame >=iterations-1:
                    ani.pause()
                return dots, prime

            # writervideo = animation.PillowWriter(fps=2, bitrate=1800)
            ani = animation.FuncAnimation(fig=fig, func=update, frames=iterations, interval = 500)
            # ani.save("gifs/wolf_latest.gif", writer = writervideo)
            plt.show()

        else:
            for i in range(iterations):
                self.move()
            return [self.wolfs[self.find_prime()[0]].x, self.find_prime()[1]]
