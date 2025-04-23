from TSP import TSP
from genetic_for_tsp import GTSP
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import MCR
from FuncLim import FuncLim
from PSO import PSOSolver

def test_mean(object, iterations, tests, desc="test_mean"):
    output = []
    for _ in tqdm(range(tests), desc=desc):
        object.reset()
        output.append([
            object.solve_stats(iterations)[2]
        ])

    return np.mean(output, axis=0)[0]


#basic GTSP test
VERTICES = 100
a = TSP(VERTICES, circle = True)
a.draw_graph()

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

a_gtsp = GTSP({
        "n_vertices": VERTICES,
        "pop_size": 50,
        "elitism": 10,
        "children": 100,
        "m_switch_prob": 0.5,
        "m_pop_prob": 0.5
    }, a.check_path, True)

a_fast_pso_gtsp = GTSP({
        "n_vertices": VERTICES,
        "pop_size": 20,
        "elitism": 5,
        "children": 100,
        "m_switch_prob": 0.44,
        "m_pop_prob": 0.92
    }, a.check_path, True)
result = MCR.MCR(a.check_path,VERTICES,10000, True)
print(result)
a.draw_graph(path=result[1])
input()
"""
y_a = test_mean(a_gtsp, 1000, 1, "a_gtsp")

plt.plot(range(len(y_a)),y_a, label="Basic test, " + str(VERTICES) + " vertices")
plt.legend()
plt.yscale("log")
plt.show()
"""
SECONDS = 5
y_gtsp = a_gtsp.solve_seconds(SECONDS)
y_pso_gtsp = a_fast_pso_gtsp.solve_seconds(SECONDS)
y_mcr = MCR.MCR_seconds(a.check_path, VERTICES, SECONDS)
plt.plot([dot[0] for dot in y_gtsp[2]],[dot[1] for dot in y_gtsp[2]], label = "gtsp")
plt.plot([dot[0] for dot in y_mcr[2]],[dot[1] for dot in y_mcr[2]], label = "mcr")
plt.plot([dot[0] for dot in y_pso_gtsp[2]],[dot[1] for dot in y_pso_gtsp[2]], label = "pso_gtsp")
plt.legend()
plt.yscale("log")
plt.show()
a.draw_graph(path = y_gtsp[1])
input()
a.draw_graph(path = y_pso_gtsp[1])
input()

a.draw_graph(path = y_mcr[1])
input()