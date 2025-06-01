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
        if np.any(value[:-1] > self.capacity):
            value = [float("inf") for v in range(len(value)-1)] + [-float("inf")]
            return value
        else:
            return value

    def __str__(self):
        table = [[],[], [], []]
        for i in range(self.capacity.shape[0]):
            table[0].append("dim " + str(i))
            table[1].append("cap:")
            table[2].append(round(self.capacity[i], 4))
            table[3].append("obj:")
        table[0].append("cost")
        table[1].append("max:")
        table[2].append(round(np.sum(self.objects[:, -1]), 4))
        table[3].append("obj:")

        table += np.round(self.objects, 4).tolist()
        table = PTC(table)
        table.transpose()
        return str(table)

def generate_random_backpack(dimensions: int, n_objects: int):
    capacity = (np.random.rand(dimensions) + 0.2) * n_objects / 2
    objects = np.random.rand(n_objects, dimensions+1)
    return Backpack(capacity, objects)