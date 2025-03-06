import numpy as np
from PSO import PSOSolver
from matplotlib import pyplot as plt
from tqdm import tqdm
from BEE import BEEHive

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

def bee_opt(pos):
    _bee = BEEHive({
        "delta": pos[0],
        "n_max": pos[1],
        "alpha": pos[2],
        "pop_size": 50,
        "l_s": int(pos[3]),
        "l_es": 10,
        "z_e": 5,
        "z_o": 2,
        "dim": 5,
        "pos_min": np.array([-5.12] * 5),
        "pos_max": np.array([5.12] * 5)
    }, rastring, True)
    result = _bee.solve(25)[0]
    return result

pso_for_bee = PSOSolver({
"a1": 0.05,#acceleration number
"a2": 0.1,#acceleration number
"pop_size": 50,#population size
"dim": 4,#dimensions
"pos_min": np.array([0,0,0,10]),#vector of minimum positions
"pos_max": np.array([1,1,1,41]),#vector of maximum positions
"speed_min": np.array([-0.5]*3+ [5]),#vector of min speed
"speed_max": np.array([0.5]*3 + [5]),#vector of max speed
}, bee_opt, True)
#print(pso_for_bee.solve(50, True))


pso = PSOSolver({
"a1": 2,#acceleration number
"a2": 3,#acceleration number
"pop_size": 25,#population size
"dim": 2,#dimensions
"pos_min": np.array([-5.12]*2),#vector of minimum positions
"pos_max": np.array([5.12]*2),#vector of maximum positions
"speed_min": np.array([-0.5]*2),#vector of min speed
"speed_max": np.array([0.5]*2),#vector of max speed
}, rastring, True)

print(pso.anisolve())

bee = BEEHive({
    "delta": 0.75,
    "n_max": 0.34,
    "alpha": 0.74,
    "pop_size": 25,
    "l_s": 20,
    "l_es": 10,
    "z_e": 5,
    "z_o": 2,
    "dim": 2,
    "pos_min":  np.array([-5.12]*2),
    "pos_max": np.array([5.12]*2)
}, rastring, True)

print(bee.solve(100))

pso_y = test_mean(pso, 100, 100)
bee_y = test_mean(bee, 100, 100)

plt.plot(range(len(pso_y)),pso_y, "b")
plt.plot(range(len(bee_y)), bee_y, "r")
plt.show()