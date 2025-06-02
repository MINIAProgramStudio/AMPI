import Backpack
from GreedyBackpackSolver import GreedyBackpackSolver as GreedyBS
from GeneticBackpackSolver import GeneticBackSolver as GeneticBS
import matplotlib.pyplot as plt

path = "data/Data50.txt"

fb = Backpack.read_backpack(path)

#print(rb)

greedy_rb = GreedyBS(fb)
greedy_choice = greedy_rb.solve(progressbar=True)
#print(greedy_choice)
#(rb.fitness(greedy_choice))


coef = {
    "pop_size": 2000,
    "children": 2000,
    "m_prob": 0.05,
}

genetic_rb = GeneticBS(fb, coef)
genetic_choice = genetic_rb.solve_stats(iterations = 100, progressbar=True)
#print(genetic_choice[0])
#print(genetic_choice[1])

plt.plot(genetic_choice[2][:,-1], label = "Genetic value")
plt.axhline(y = fb.fitness(greedy_choice)[-1], c = "r", label = "Greedy value")
plt.legend()
plt.show()

plt.plot(genetic_choice[2][:,:-1]*100/fb.capacity, label = "Genetic weight %")
for y_line in fb.fitness(greedy_choice)[:-1]*100/fb.capacity:
    plt.axhline(y = y_line, label = "Greedy weight %")
plt.legend()
plt.show()