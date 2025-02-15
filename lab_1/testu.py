import wolfu as wolfu
import numpy as np

def rastring(pos):
    return 10*len(pos) + sum([x**2 - 10*np.cos(2*np.pi*x) for x in pos])

packu = wolfu.PackU([
    [-5.12, 5.12],
[-5.12, 5.12],
[-5.12, 5.12],
[-5.12, 5.12],
[-5.12, 5.12],
], rastring, 100, seeking_min=True)

print(packu.solve(1000, True))