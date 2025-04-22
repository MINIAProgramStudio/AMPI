from TSP import TSP
from genetic_for_tsp import GTSP

#basic GTSP test
VERTICES = 10
a = TSP(VERTICES, circle = True)
a.draw_graph()

a_gtsp = GTSP({
    "n_vertices": VERTICES,
    "pop_size": 5,
    "children": 10,
    "m_switch_prob": 0.1,
    "m_pop_prob": 0.1
}, a.check_path)

result = a_gtsp.solve(10, True)
a.draw_graph(path=result[0])
input()