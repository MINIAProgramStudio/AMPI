import wolf_one_dimension as wond
import genetic_one_dimention as gond
import math

def func(x):
    return x**3-50*x**2

x_min = 0
x_max = 75
iterations = 25

WolfPack = wond.OneDimentionalPackOfWolfs(x_min,x_max, func, 4, seeking_min=True)
start_best = WolfPack.find_prime()[1]
WolfPack.solve(animate=True, iterations=iterations)
wolf_end_best = WolfPack.find_prime()[1]

population = gond.OneDimentionalGeneticSolver(x_min,x_max, func, 4,4, seeking_min=True)
genetic_end_best = population.solve(iterations, True)

print("Best value by random points: "+str(start_best))
print("Best by wolfpack: "+str(wolf_end_best))
print("Best by genetic: "+str(genetic_end_best))
print("Wolfpack better than random points by "+str(abs(wolf_end_best/start_best+0.1)*100-100)+"%")

