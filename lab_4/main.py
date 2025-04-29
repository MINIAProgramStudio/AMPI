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

"""
VERTICES = 30
a = TSP(VERTICES, circle = False, init_progressbar=True)
a.draw_graph()

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
    "a": 1,
    "b": 2,
    "evaporation": 0.1,
    "Q": 5
}, a)

SECONDS = 5
y_ants = a_ants.solve_seconds(SECONDS)
a.draw_graph(path=y_ants[1])
print("ants_finished")
y_medium_pso_gtsp = a_medium_pso_gtsp.solve_seconds(SECONDS)
print("mpso_finished")
y_mcr = MCR.MCR_seconds(a.check_path, VERTICES, SECONDS)
print("mcr_finished")
y_greed_gtsp = a_greed_gtsp.solve_seconds(SECONDS)
print("greed_finished")

plt.plot([dot[0] for dot in y_mcr[2]],[dot[1] for dot in y_mcr[2]], label = "mcr")
plt.plot([dot[0] for dot in y_medium_pso_gtsp[2]],[dot[1] for dot in y_medium_pso_gtsp[2]], label = "pso_medium_gtsp")
plt.plot([dot[0] for dot in y_greed_gtsp[2]],[dot[1] for dot in y_greed_gtsp[2]], label = "greedy_gtsp")
plt.plot([dot[0] for dot in y_ants[2]],[dot[1] for dot in y_ants[2]], label = "ants")
plt.legend()
plt.yscale("log")
plt.ylabel("best_value")
plt.xlabel("time (s)")
plt.show()

VERTICES = 50
a = TSP(VERTICES, circle = False, init_progressbar=True)
a.draw_graph()

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
    "a": 1,
    "b": 2,
    "evaporation": 0.25,
    "Q": 0.1,
    "ants_step": 5
}, a)

SECONDS = 5
y_ants = a_ants.solve_seconds(SECONDS)
a.draw_graph(path=y_ants[1])
y_medium_pso_gtsp = a_medium_pso_gtsp.solve_seconds(SECONDS)
y_mcr = MCR.MCR_seconds(a.check_path, VERTICES, SECONDS)
y_greed_gtsp = a_greed_gtsp.solve_seconds(SECONDS)

plt.plot([dot[0] for dot in y_mcr[2]],[dot[1] for dot in y_mcr[2]], label = "mcr")
plt.plot([dot[0] for dot in y_medium_pso_gtsp[2]],[dot[1] for dot in y_medium_pso_gtsp[2]], label = "pso_medium_gtsp")
plt.plot([dot[0] for dot in y_greed_gtsp[2]],[dot[1] for dot in y_greed_gtsp[2]], label = "greedy_gtsp")
plt.plot([dot[0] for dot in y_ants[2]],[dot[1] for dot in y_ants[2]], label = "ants")
plt.legend()
plt.yscale("log")
plt.ylabel("best_value")
plt.xlabel("time (s)")
plt.show()

VERTICES = 100
a = TSP(VERTICES, circle = False, init_progressbar=True)
a.draw_graph()

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
    "a": 1,
    "b": 2,
    "evaporation": 0.25,
    "Q": 0.1,
    "ants_step": 10
}, a)

SECONDS = 10
y_ants = a_ants.solve_seconds(SECONDS)
a.draw_graph(path=y_ants[1])
y_medium_pso_gtsp = a_medium_pso_gtsp.solve_seconds(SECONDS)
y_mcr = MCR.MCR_seconds(a.check_path, VERTICES, SECONDS)
y_greed_gtsp = a_greed_gtsp.solve_seconds(SECONDS)

plt.plot([dot[0] for dot in y_mcr[2]],[dot[1] for dot in y_mcr[2]], label = "mcr")
plt.plot([dot[0] for dot in y_medium_pso_gtsp[2]],[dot[1] for dot in y_medium_pso_gtsp[2]], label = "pso_medium_gtsp")
plt.plot([dot[0] for dot in y_greed_gtsp[2]],[dot[1] for dot in y_greed_gtsp[2]], label = "greedy_gtsp")
plt.plot([dot[0] for dot in y_ants[2]],[dot[1] for dot in y_ants[2]], label = "ants")
plt.legend()
plt.yscale("log")
plt.ylabel("best_value")
plt.xlabel("time (s)")
plt.show()

VERTICES = 200
a = TSP(VERTICES, circle = False, init_progressbar=True)
a.draw_graph()

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
    "a": 1,
    "b": 2,
    "evaporation": 0.01,
    "Q": 15,
    "ants_step": 10
}, a)

SECONDS = 10
y_ants = a_ants.solve_seconds(SECONDS)
a.draw_graph(path=y_ants[1])
y_mcr = MCR.MCR_seconds(a.check_path, VERTICES, SECONDS)
y_greed_gtsp = a_greed_gtsp.solve_seconds(SECONDS)
y_medium_pso_gtsp = a_medium_pso_gtsp.solve_seconds(SECONDS)
plt.plot([dot[0] for dot in y_mcr[2]],[dot[1] for dot in y_mcr[2]], label = "mcr")
plt.plot([dot[0] for dot in y_medium_pso_gtsp[2]],[dot[1] for dot in y_medium_pso_gtsp[2]], label = "pso_medium_gtsp")
plt.plot([dot[0] for dot in y_greed_gtsp[2]],[dot[1] for dot in y_greed_gtsp[2]], label = "greedy_gtsp")
plt.plot([dot[0] for dot in y_ants[2]],[dot[1] for dot in y_ants[2]], label = "ants")
plt.legend()
plt.yscale("log")
plt.ylabel("best_value")
plt.xlabel("time (s)")
plt.show()

VERTICES = 200
a = TSP(VERTICES, circle = True, init_progressbar=True)
a.draw_graph()

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
    "a": 1,
    "b": 2,
    "evaporation": 0.25,
    "Q": 0.01,
    "ants_step": 1
}, a)

SECONDS = 10
y_ants = a_ants.solve_seconds(SECONDS)
a.draw_graph(path=y_ants[1])
y_medium_pso_gtsp = a_medium_pso_gtsp.solve_seconds(SECONDS)
y_mcr = MCR.MCR_seconds(a.check_path, VERTICES, SECONDS)
y_greed_gtsp = a_greed_gtsp.solve_seconds(SECONDS)

plt.plot([dot[0] for dot in y_mcr[2]],[dot[1] for dot in y_mcr[2]], label = "mcr")
plt.plot([dot[0] for dot in y_medium_pso_gtsp[2]],[dot[1] for dot in y_medium_pso_gtsp[2]], label = "pso_medium_gtsp")
plt.plot([dot[0] for dot in y_greed_gtsp[2]],[dot[1] for dot in y_greed_gtsp[2]], label = "greedy_gtsp")
plt.plot([dot[0] for dot in y_ants[2]],[dot[1] for dot in y_ants[2]], label = "ants")
plt.legend()
plt.yscale("log")
plt.ylabel("best_value")
plt.xlabel("time (s)")
plt.show()
"""
VERTICES = 600
a = TSP(VERTICES, circle = False, init_progressbar=True)
a.draw_graph()

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
    "a": 1,
    "b": 10,
    "evaporation": 0.01,
    "Q": 30,
    "ants_step": 100
}, a)

SECONDS = 15
y_ants = a_ants.solve_seconds(SECONDS)
a.draw_graph(path=y_ants[1])
#y_medium_pso_gtsp = a_medium_pso_gtsp.solve_seconds(SECONDS)
#y_mcr = MCR.MCR_seconds(a.check_path, VERTICES, SECONDS)
y_greed_gtsp = a_greed_gtsp.solve_seconds(SECONDS)

#plt.plot([dot[0] for dot in y_mcr[2]],[dot[1] for dot in y_mcr[2]], label = "mcr")
#plt.plot([dot[0] for dot in y_medium_pso_gtsp[2]],[dot[1] for dot in y_medium_pso_gtsp[2]], label = "pso_medium_gtsp")
plt.plot([dot[0] for dot in y_greed_gtsp[2]],[dot[1] for dot in y_greed_gtsp[2]], label = "greedy_gtsp")
plt.plot([dot[0] for dot in y_ants[2]],[dot[1] for dot in y_ants[2]], label = "ants")
plt.legend()
plt.yscale("log")
plt.ylabel("best_value")
plt.xlabel("time (s)")
plt.show()