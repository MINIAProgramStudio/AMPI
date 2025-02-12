import wolf_one_dimension as wod
import math

def func(x):
    return x**3-20*x**2

WolfPack = wod.OneDimentionalPackOfWolfs(0,20, func, 3, seeking_min=True)
print(WolfPack.find_prime())
WolfPack.solve(animate=True, iterations=20)
print(WolfPack.find_prime())