import numpy as np
from random import random
from tqdm import tqdm
import copy
import time

def MCR(func, vertices, attempts, progressbar = False):
    iterator = range(attempts)
    if progressbar:
        iterator = tqdm(iterator)
    best = np.random.permutation(vertices)
    best_val = func(best)
    for _ in iterator:
        r = np.random.permutation(vertices)
        if func(r) < best_val:
            best = r
            best_val = func(best)
    return (best_val, best.tolist())

def MCR_seconds(func, vertices, seconds):
    output = []
    start = time.time()
    best = np.random.permutation(vertices)
    best_val = func(best)
    while start + seconds > time.time():
        r = np.random.permutation(vertices)
        if func(r) < best_val:
            best = r
            best_val = func(best)
            output.append([time.time() - start, best_val])
    output.append([time.time() - start, best_val])
    return (best_val, best.tolist(), output)