import wolf_one_dimension as wond
import genetic_one_dimention as gond
import wolf_two_dimentions as wolf2d
import math




"""
def od_func(x):
    return x**3-50*x**2
    
    
x_min = 0
x_max = 75
iterations = 25

WolfPack = wond.OneDimensionalPackOfWolfs(x_min,x_max, od_func, 4, seeking_min=True)
start_best = WolfPack.find_prime()[1]
WolfPack.solve(animate=True, iterations=iterations)
wolf_end_best = WolfPack.find_prime()[1]

population = gond.OneDimensionalGeneticSolver(x_min,x_max, od_func, 4,4, seeking_min=True)
genetic_end_best = population.solve(iterations, True)

print("Best value by random points: "+str(start_best))
print("Best by wolfpack: "+str(wolf_end_best))
print("Best by genetic: "+str(genetic_end_best))
print("Wolfpack better than random points by "+str(abs(wolf_end_best/start_best+0.1)*100-100)+"%")
"""

def func_2d(x,y):
    return x**2 + y**2

x_min = -5
x_max = 5
y_min = -5
y_max = 5
WolfPack2D = wolf2d.TwoDimensionalPackOfWolfs(x_min, x_max, y_min, y_max, func_2d,10, seeking_min=True)

WolfPack2D.solve(100, True)