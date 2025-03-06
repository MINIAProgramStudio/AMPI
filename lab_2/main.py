import numpy as np
from PSO import PSOSolver
from matplotlib import pyplot as plt
from tqdm import tqdm


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

pso = PSOSolver({
"a1": 2,#acceleration number
"a2": 3,#acceleration number
"pop_size": 50,#population size
"dim": 5,#dimensions
"pos_min": np.array([-5.12]*5),#vector of minimum positions
"pos_max": np.array([5.12]*5),#vector of maximum positions
"speed_min": np.array([-0.5]*5),#vector of min speed
"speed_max": np.array([0.5]*5),#vector of max speed
}, rastring, True)

print(pso.anisolve())

y = test_mean(pso, 1000, 100)
x = range(len(y))
plt.plot(x,y)
plt.show()