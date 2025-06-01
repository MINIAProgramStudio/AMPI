import Backpack
from GreedyBackpackSolver import GreedyBackpackSolver as GBS

rb = Backpack.generate_random_backpack(1,10000)

#print(rb)

gbs_rb = GBS(rb)
gbs_choice = gbs_rb.solve(progressbar=True)
print(gbs_choice)
print(rb.fitness(gbs_choice))