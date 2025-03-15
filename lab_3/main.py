from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from CC import  CC
from PSO import PSOSolver

def test_mean(object, iterations, tests, desc = "test_mean"):
    output = []
    for _ in tqdm(range(tests), desc = desc):
        object.reset()
        output.append([
            object.solve_stats(iterations)[2]
        ])
    return np.mean(output, axis=0)[0]

def rastring(pos):
    output = 10*len(pos)
    for d in range(len(pos)):
        output += pos[d]**2 - 10*np.cos(2*np.pi*pos[d])
    return output

def cc_optimiser(pos):
    _cc = CC({
    "p_detect": pos[0],
    "delta": pos[1],


    "pop_size": 25,
    "dim": 20,
    "pos_min": np.array([-5.12] * 20),
    "pos_max": np.array([5.12] * 20)
}, rastring, True)
    return _cc.solve(1000, False)[0]
"""
pso_for_cc = PSOSolver({
"a1": 0.005,#acceleration number
"a2": 0.01,#acceleration number
"pop_size": 10,#population size
"dim": 2,#dimensions
"pos_min": np.array([0,0]),#vector of minimum positions
"pos_max": np.array([1,1]),#vector of maximum positions
"speed_min": np.array([-0.5]*2),#vector of min speed
"speed_max": np.array([0.5]*2),#vector of max speed
}, cc_optimiser, True)
print(pso_for_cc.solve(100, True))
"""
cc_rastring = CC({
    "p_detect": 0.02,
    "delta": 0.0025,


    "pop_size": 10,
    "dim": 20,
    "pos_min": np.array([-5.12] * 20),
    "pos_max": np.array([5.12] * 20)
}, rastring, True)

cc_y = test_mean(cc_rastring, 10000, 20)
plt.plot(range(len(cc_y)),cc_y, "b", label = "CC")
plt.legend()
plt.show()