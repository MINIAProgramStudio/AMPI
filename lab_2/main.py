import numpy as np
from PSO import PSOSolver


def rastring(pos):
    output = 10*len(pos)
    for d in range(len(pos)):
        output += pos[d]**2 - 10*np.cos(2*np.pi*pos[d])
    return output

pso = PSOSolver({
"a1": 0.001,#acceleration number
"a2": 0.001,#acceleration number
"max_iter": 1000,#max iterations
"pop_size": 1000,#population size
"dim": 2,#dimensions
"pos_min": np.array([-5.12]*2),#vector of minimum positions
"pos_max": np.array([5.12]*2),#vector of maximum positions
"speed_min": np.array([0]*2),#vector of min speed
"speed_max": np.array([2**16]*2),#vector of max speed
}, rastring, True)

print(pso.solve())