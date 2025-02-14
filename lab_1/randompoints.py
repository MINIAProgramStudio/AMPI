from random import random

def randomsolver(minmax, func, probes, seeking_min = False):
    dimensions = len(minmax)
    args = []
    value = 0
    if seeking_min:
        best = float("inf")
    else:
        best = -float("inf")
    for p in range(probes):
        args = [random()*(minmax[i][1]-minmax[i][0])+minmax[i][0] for i in range(dimensions)]
        value = func(*args)
        if seeking_min:
            if value<best:
                best = value
        else:
            if value > best:
                best = value
    return best

