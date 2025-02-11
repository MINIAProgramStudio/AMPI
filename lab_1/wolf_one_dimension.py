from random import random

class OneDimentionalWolf:
    def __init__(self, x, speed):
        self.x = x
        self.speed = speed

class OneDimentionalPackOfWolfs:
    def __init__(self, x_min, x_max, func, n_of_wolfs, speed_func, seeking_min = False):
        self.x_min = x_min
        self.x_max = x_max
        self.func = func
        self.speed_func = speed_func
        self.seeking_min = seeking_min
        if n_of_wolfs >= 2:
            self.wolfs = [OneDimentionalWolf(x_min, self.speed_func(random())), # Вовк на мінімальному значенні
                          OneDimentionalWolf(x_max, self.speed_func(random())) # Вовк на максимальному значенні
                          ]
            if n_of_wolfs >=3:
                for i in range(2,n_of_wolfs):
                    self.wolfs.append(OneDimentionalWolf(random()*(x_max-x_min)+x_min, self.speed_func(random))) # Вовки на випадкових позиціях
        else:
            raise Exception("too few wolfs")

    def find_prime(self):
        prime = 0
        if self.seeking_min:
            prime_value = float("inf")
            for i in range(len(self.wolfs)):
                value = self.func(self.wolfs[i])
                if value < prime_value:
                    prime = i
                    prime_value = value
        else:
            prime_value = -float("inf")
            for i in range(len(self.wolfs)):
                value = self.func(self.wolfs[i])
                if value > prime_value:
                    prime = i
                    prime_value = value
        return [prime, prime_value]

    def move(self):
        prime = self.find_prime()[0]
        for i in range(len(self.wolfs)):
            if i != prime:
                if self.wolfs[i].x < self.wolfs[prime]:
                    self.wolfs[i].x += self.wolfs[i].speed
                else:
                    self.wolfs[i].x -= self.wolfs[i].speed

    