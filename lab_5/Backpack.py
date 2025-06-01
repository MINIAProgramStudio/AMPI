import numpy as np
from PythonTableConsole import PythonTableConsole as PTC

class Backpack:
    def __init__(self, capacity: np.array, objects: np.array):
        if not capacity.shape[0] + 1 == objects.shape[1]:
            raise Exception("E Backpack: capacity and objects shape mismatch")
        self.capacity = capacity
        self.objects = objects

    def fitness(self, choice):
        if not self.objects.shape[0] == choice.shape[0]:
            raise Exception("E Backpack: choice and objects shape mismatch")
        selected_objects = self.objects[np.where(choice == 1)]
        value = np.sum(selected_objects, axis = 0)
        if np.any(value > self.capacity):
            return np.zeros_like(self.capacity)
        else:
            return value

    def __str__(self):
        table = [[]]
        for i in range(self.capacity.shape[0]):
            table[0].append("dim " + str(i))
        table[0].append("cost")
        table += np.round(self.objects, 4).tolist()
        table = PTC(table)
        table.transpose()
        return str(table)

def generate_random_backpack(dimensions: int, n_objects: int):
    capacity = np.random.rand(dimensions) * n_objects
    objects = np.random.rand(n_objects, dimensions+1)
    return Backpack(capacity, objects)