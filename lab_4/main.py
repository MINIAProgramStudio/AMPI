from TSP import TSP
from genetic_for_tsp import GTSP
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import MCR
from FuncLim import FuncLim
from PSO import PSOSolver
from ants import AntSolver
import time

def test_mean(object, iterations, tests, desc="test_mean"):
    output = []
    for _ in tqdm(range(tests), desc=desc):
        object.reset()
        output.append([
            object.solve_stats(iterations)[2]
        ])

    return np.mean(output, axis=0)[0]


#basic GTSP test
VERTICES = 30
a = TSP(VERTICES, circle = True, init_progressbar=True)
a.draw_graph()
"""
def GTSP_optimiser(pos):

    _a_gtsp = GTSP({
        "n_vertices": VERTICES,
        "pop_size": int(250**pos[0]),
        "elitism": int(200**pos[1]),
        "children": int(500**pos[2]),
        "m_switch_prob": pos[3],
        "m_pop_prob": pos[4]
    }, a.check_path, True)
    result = _a_gtsp.solve_seconds(1.5)
    return result[0]

def GTSP_optimiser_limit_1(pos):
    return pos[0]-pos[1]

GTSP_optimiser_limits = FuncLim(GTSP_optimiser, [[GTSP_optimiser_limit_1,10,1]])

PSO_GTSP = PSOSolver({
    "a1": 0.1,#acceleration number
    "a2": 0.2,#acceleration number
    "pop_size": 15,#population size
    "dim": 5,#dimensions
    "pos_min": np.array([0,0,0,0,0]),#vector of minimum positions
    "pos_max": np.array([1,1,1,1,1]),#vector of maximum positions
    "speed_min": np.array([-0.1,-0.1,-0.1,-0.1,-0.1]),#vector of min speed
    "speed_max": np.array([0.1,0.1,0.1,0.1,0.1]),#vector of max speed
}, GTSP_optimiser_limits.func, seeking_min = True)
pso_result = PSO_GTSP.solve(30,True)
print("pso",pso_result[0],pso_result[1])
input()


def ants_optimiser(pos):
    _a_ants = AntSolver({
        "a": pos[0],
        "b": pos[1],
        "evaporation": pos[2],
        "Q": pos[3]
    }, a)
    result = _a_ants.solve_seconds(2)
    return result[0]


PSO_ants = PSOSolver({
    "a1": 0.4,#acceleration number
    "a2": 0.8,#acceleration number
    "pop_size": 5,#population size
    "dim": 4,#dimensions
    "pos_min": np.array([0,0,0,0]),#vector of minimum positions
    "pos_max": np.array([10,10,10,30]),#vector of maximum positions
    "speed_min": np.array([-5,-5,-5,-5]),#vector of min speed
    "speed_max": np.array([5,5,5,5]),#vector of max speed
}, ants_optimiser, seeking_min=True)
pso_result = PSO_ants.solve(30,True)
print("pso", pso_result[0], pso_result[1])
"""

a_gtsp = GTSP({
        "n_vertices": VERTICES,
        "pop_size": 50,
        "elitism": 10,
        "children": 100,
        "m_switch_prob": 0.5,
        "m_pop_prob": 0.5,
        "greed": False,
    }, a.check_path, True)

a_fast_pso_gtsp = GTSP({
        "n_vertices": VERTICES,
        "pop_size": 20,
        "elitism": 5,
        "children": 100,
        "m_switch_prob": 0.44,
        "m_pop_prob": 0.92,
        "greed": False,
    }, a.check_path, True)

a_medium_pso_gtsp = GTSP({
        "n_vertices": VERTICES,
        "pop_size": 220,
        "elitism": 100,
        "children": 10,
        "m_switch_prob": 0.72,
        "m_pop_prob": 0.53,
        "greed": False,
    }, a.check_path, True)

a_greed_gtsp = GTSP({
        "n_vertices": VERTICES,
        "pop_size": min(50,VERTICES),
        "elitism": 10,
        "children": 100,
        "m_switch_prob": 0.9,
        "m_pop_prob": 0.9,
        "greed": a.matrix,
}, a.check_path, True)

a_ants = AntSolver({
    "a": 0.5,
    "b": 2,
    "evaporation": 0.1,
    "Q": 10
}, a)
a_ants.reset()
"""
a_ants.anisolve(a, iterations=100, step=10)
input()
"""
y_a = test_mean(a_ants, 100, 5, "ants")

plt.plot(range(len(y_a)),y_a, label="Basic test, " + str(VERTICES) + " vertices")
plt.legend()
plt.yscale("log")
plt.show()

SECONDS = 10
#y_gtsp = a_gtsp.solve_seconds(SECONDS)
#y_fast_pso_gtsp = a_fast_pso_gtsp.solve_seconds(SECONDS)
#y_medium_pso_gtsp = a_medium_pso_gtsp.solve_seconds(SECONDS)
#y_mcr = MCR.MCR_seconds(a.check_path, VERTICES, SECONDS)
y_greed_gtsp = a_greed_gtsp.solve_seconds(SECONDS)
y_ants = a_ants.solve_seconds(SECONDS)
#plt.plot([dot[0] for dot in y_gtsp[2]],[dot[1] for dot in y_gtsp[2]], label = "gtsp")
#plt.plot([dot[0] for dot in y_mcr[2]],[dot[1] for dot in y_mcr[2]], label = "mcr")
#plt.plot([dot[0] for dot in y_fast_pso_gtsp[2]],[dot[1] for dot in y_fast_pso_gtsp[2]], label = "pso_fast_gtsp")
#plt.plot([dot[0] for dot in y_medium_pso_gtsp[2]],[dot[1] for dot in y_medium_pso_gtsp[2]], label = "pso_medium_gtsp")
plt.plot([dot[0] for dot in y_greed_gtsp[2]],[dot[1] for dot in y_greed_gtsp[2]], label = "greedy_gtsp")
plt.plot([dot[0] for dot in y_ants[2]],[dot[1] for dot in y_ants[2]], label = "ants")
plt.legend()
plt.yscale("log")
plt.ylabel("best_value")
plt.xlabel("time (s)")
plt.show()

a.draw_graph(path = y_gtsp[1])
input()
a.draw_graph(path = y_fast_pso_gtsp[1])
input()
a.draw_graph(path = y_medium_pso_gtsp[1])
input()
#a.draw_graph(path = y_mcr[1])
#input()
""""""