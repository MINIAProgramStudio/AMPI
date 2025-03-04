import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation, cm

class PSOParticle:
    def __init__(self, coef, func, seeking_min):
        self.c = coef
        self.pos = np.random.random_sample(self.c["dim"]) * (self.c["pos_max"]-self.c["pos_min"]) + self.c["pos_min"]
        self.speed = np.random.random_sample(self.c["dim"]) * (self.c["speed_max"]-self.c["speed_min"]) + self.c["speed_min"]
        self.best_pos = self.pos
        self.func = func
        self.seeking_min = seeking_min
        self.best_val = self.func(self.pos)

    def update(self, global_best_pos, r1, r2):
        # оновлення швидкості
        self.speed = self.speed + self.c["a1"] * (self.best_pos - self.pos) * r1 + self.c["a2"] * (global_best_pos - self.pos) * r2
        self.speed = np.minimum(self.c["speed_max"], np.maximum(self.c["speed_min"], self.speed))
        # оновлення позиції
        self.pos = self.pos + self.speed
        for d in range(self.c["dim"]):
            if self.pos[d] < self.c["pos_min"][d]:
                self.pos[d] = self.c["pos_min"][d] + abs(self.pos[d]-self.c["pos_min"][d])
            if self.pos[d] > self.c["pos_max"][d]:
                self.pos[d] = self.c["pos_max"][d] - abs(self.pos[d] - self.c["pos_max"][d])
        # оновлення найкращої позиції
        result = self.func(self.pos)
        if ((result > self.best_val) and not self.seeking_min) or ((result < self.best_val) and self.seeking_min) :
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
                if ((result[1] > self.best_val) and not self.seeking_min) or ((result[1] < self.best_val) and self.seeking_min):
                    new_best_val = result[1]
                    new_best_pos = result[0]
        self.best_val = new_best_val
        self.best_pos = new_best_pos

    def solve(self, animate = False, target = 0):
        if animate:
            if self.c["dim"] == 2:
                fig = plt.figure()
                ax = plt.axes(projection="3d", computed_zorder=False)
                dots_x = [particle.pos[0] for particle in self.pop]
                dots_y = [particle.pos[1] for particle in self.pop]
                dots_z = [self.func(particle.pos) for particle in self.pop]
                prime_x = [self.best_pos[0]]
                prime_y = [self.best_pos[1]]
                prime_z = [self.best_val]

                func_x = np.linspace(self.c["pos_min"][0], self.c["pos_max"][0], 100)
                func_y = np.linspace(self.c["pos_min"][1], self.c["pos_max"][1], 100)
                FUNC_X, FUNC_Y = np.meshgrid(func_x, func_y)

                FUNC_Z = self.func([FUNC_X, FUNC_Y])

                surface = ax.plot_surface(FUNC_X, FUNC_Y, FUNC_Z, cmap=cm.coolwarm,
                                          linewidth=0, antialiased=False, zorder=0)
                dots = ax.scatter(dots_x, dots_y, dots_z, c="#ff0000", zorder=5)
                prime = ax.scatter(prime_x, prime_y, prime_z, s=75, c="#ffff00", zorder=10)

                fig.suptitle("PSO " + str(0) + "/" + str(self.c["max_iter"]) + " Best: " + str(round(self.best_val, 3)))

                def update(frame):
                    self.iter()
                    dots_x = [particle.pos[0] for particle in self.pop]
                    dots_y = [particle.pos[1] for particle in self.pop]
                    dots_z = [self.func(particle.pos) for particle in self.pop]
                    prime_x = [self.best_pos[0]]
                    prime_y = [self.best_pos[1]]
                    prime_z = [self.best_val]

                    dots.set_offsets(np.c_[dots_x, dots_y])
                    dots.set_3d_properties(dots_z, zdir='z')
                    prime.set_offsets(np.c_[prime_x, prime_y])
                    prime.set_3d_properties(prime_z, zdir='z')
                    fig.suptitle(
                        "PSO " + str(frame + 1) + "/" + str(self.c["max_iter"]) + " Best: " + str(round(self.best_val, 3)))
                    if frame >= self.c["max_iter"] - 1:
                        ani.pause()
                    return dots, prime

                # writervideo = animation.PillowWriter(fps=2, bitrate=1800)
                ani = animation.FuncAnimation(fig=fig, func=update, frames=self.c["max_iter"], interval=50)
                # ani.save("gifs/wolf_latest.gif", writer = writervideo)
                plt.show()
                return (self.best_val, self.best_pos)
            else:
                px = list(range(self.c["max_iter"] + 1))
                py = [abs(self.best_val - target)]
                fig, ax = plt.subplots()
                for i in range(self.c["max_iter"]):
                    self.iter()
                    py.append(abs(self.best_val - target))

                ax.set_yscale("log")

                graph = ax.plot(px, py)[0]
                plt.show()
                return (self.best_val, self.best_pos)
        else:
            for iter in tqdm(range(self.c["max_iter"]), desc="PSO"):
                self.iter()
            return (self.best_val, self.best_pos)