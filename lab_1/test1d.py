from random import random

import randompoints as rp
import wolf1d as wolf1d
import genetic1d as genetic1d
import wolf2d
import math
import PythonTableConsole as PTC
import numpy as np
from time import time
from tqdm import tqdm

result_table = PTC.PythonTableConsole([
    ["No", "Функція","Метод пошуку","Параметри","Час створення","Час пошуку", "Загальний час","Результат в середньому", "Найкращий можливий результат", "Відносна середня похибка, %"]
])

# Тестування одновимірних функцій

repeat1d = 10**3

def garmonic(x):
    return x**3 * (3-x)**5 * math.sin(10*math.pi*x)
x_min = 0
x_max = 3

# метод випадкових точок
random_points_number = 450
random_points_best_value = 0
random_time = 0

for i in tqdm(range(repeat1d), desc="random1d"):
    start = time()

    random_points_best_value += rp.randomsolver([[x_min, x_max]], garmonic, random_points_number, seeking_min=True)
    stop = time()
    random_time += stop-start

result_table.contains.append([
    1,
    "Гармонічна",
    "Випадкові точки",
    "X є ["+str(x_min)+"; "+str(x_max)+"], Точок: " + str(random_points_number),
    0,
    str(round(random_time*10**6/repeat1d, 1)) + "mcs",
    str(round(random_time*10**6/repeat1d, 1)) + "mcs",
    round(random_points_best_value/repeat1d, 3),
    -32.957,
    round(abs(100-random_points_best_value*100/(-32.957*repeat1d)),2)
])
"""
# метод вовків, мало вовків, багато ітерацій
wolf_number = 10
wolf_iterations = 41
wolf_best_value = 0
wolf_solve_time = 0
wolf_startup_time=0

for i in tqdm(range(repeat1d), desc="wolf1d"):
    start = time()
    WolfPack1d = wolf1d.Pack1D(x_min, x_max, garmonic, wolf_number, seeking_min=True)
    stop = time()
    wolf_startup_time += stop - start
    start = time()
    WolfPack1d.solve(wolf_iterations)
    wolf_best_value += WolfPack1d.find_prime()[1]
    stop = time()
    wolf_solve_time += stop - start

result_table.contains.append([
    result_table.contains[-1][0]+1,
    "Гармонічна",
    "Вовки<Ітерації",
    "X є ["+str(x_min)+"; "+str(x_max)+"], Вовків: " + str(wolf_number) + ", Ітерацій: " + str(wolf_iterations),
    str(round(wolf_startup_time*10**6/repeat1d, 1)) + "mcs",
    str(round(wolf_solve_time*10**6/repeat1d, 1)) + "mcs",
    str(round((wolf_startup_time + wolf_solve_time)*10**6/repeat1d, 1)) + "mcs",
    round(wolf_best_value/repeat1d, 3),
    -32.957,
    round(abs(100-wolf_best_value*100/(-32.957*repeat1d)),2)
])
"""
# метод вовків, врівноважений
wolf_number = 20
wolf_iterations = 20
wolf_best_value = 0
wolf_solve_time = 0
wolf_startup_time=0

for i in tqdm(range(repeat1d), desc="wolf1d"):
    start = time()
    WolfPack1d = wolf1d.Pack1D(x_min,x_max,garmonic,wolf_number, seeking_min=True)
    stop = time()
    wolf_startup_time += stop-start
    start = time()
    WolfPack1d.solve(wolf_iterations)
    wolf_best_value += WolfPack1d.find_prime()[1]
    stop = time()
    wolf_solve_time += stop-start

result_table.contains.append([
    result_table.contains[-1][0]+1,
    "Гармонічна",
    "Вовки=Ітерації",
    "X є ["+str(x_min)+"; "+str(x_max)+"], Вовків: " + str(wolf_number) + ", Ітерацій: " + str(wolf_iterations),
    str(round(wolf_startup_time * 10 ** 6 / repeat1d, 1)) + "mcs",
    str(round(wolf_solve_time * 10 ** 6 / repeat1d, 1)) + "mcs",
    str(round((wolf_startup_time + wolf_solve_time) * 10 ** 6 / repeat1d, 1)) + "mcs",
    round(wolf_best_value/repeat1d, 3),
    -32.957,
    round(abs(100-wolf_best_value*100/(-32.957*repeat1d)),2)
])
"""
# метод вовків, багато вовків, мало ітерацій
wolf_number = 37
wolf_iterations = 10
wolf_best_value = 0
wolf_solve_time = 0
wolf_startup_time=0

for i in tqdm(range(repeat1d), desc="wolf1d"):
    start = time()
    WolfPack1d = wolf1d.Pack1D(x_min, x_max, garmonic, wolf_number, seeking_min=True)
    stop = time()
    wolf_startup_time += stop - start
    start = time()
    WolfPack1d.solve(wolf_iterations)
    wolf_best_value += WolfPack1d.find_prime()[1]
    stop = time()
    wolf_solve_time += stop - start

result_table.contains.append([
    result_table.contains[-1][0]+1,
    "Гармонічна",
    "Вовки>Ітерації",
    "X є ["+str(x_min)+"; "+str(x_max)+"], Вовків: " + str(wolf_number) + ", Ітерацій: " + str(wolf_iterations),
    str(round(wolf_startup_time * 10 ** 6 / repeat1d, 1)) + "mcs",
    str(round(wolf_solve_time * 10 ** 6 / repeat1d, 1)) + "mcs",
    str(round((wolf_startup_time + wolf_solve_time) * 10 ** 6 / repeat1d, 1)) + "mcs",
    round(wolf_best_value/repeat1d, 3),
    -32.957,
    round(abs(100-wolf_best_value*100/(-32.957*repeat1d)),2)
])
"""

# генетичний метод
genetic_pop_size = 15
genetic_bits = 16
genetic_iterations = 20
genetic_best_value = 0
genetic_solve_time = 0
genetic_startup_time = 0
for i in tqdm(range(repeat1d)):
#for i in tqdm(range(repeat1d), desc="genetic1d"):
    start = time()
    geneticsolver1d = genetic1d.Genetic1D(x_min, x_max, garmonic, genetic_pop_size, genetic_pop_size, genetic_bits, seeking_min=True)
    stop = time()
    genetic_startup_time += stop-start
    start = time()
    genetic_best_value += geneticsolver1d.solve(genetic_iterations)
    stop = time()
    genetic_solve_time += stop-start

result_table.contains.append([
    result_table.contains[-1][0]+1,
    #result_table.contains[-1][0]+1,
    "Гармонічна",
    "Генетичний",
    "X є ["+str(x_min)+"; "+str(x_max)+"], Поп.: " + str(genetic_pop_size) + ", Біт: " + str(genetic_bits)+ ", Іт.: "+str(genetic_iterations),
    str(round(genetic_startup_time*10**6/repeat1d, 1)) + "mсs",
    str(round(genetic_solve_time*10**6/repeat1d, 1)) + "mсs",
    str(round((genetic_solve_time+genetic_startup_time)*10**6/repeat1d, 1)) + "mсs",
    round(genetic_best_value/repeat1d, 3),
    -32.957,
    round(abs(100-genetic_best_value*100/(-32.957*repeat1d)),2)
])




x_min = -1 #a
x_max = 1 #b
p = 2
q = 1
m = 3
n = 2

def parametric(x):
    return (x_min - x)**m + (x_max-x)**n * math.sin(math.pi*p*x) * math.sin(math.pi*q*x)

# метод випадкових точок
random_points_number = 450
random_points_best_value = 0
random_time = 0

for i in tqdm(range(repeat1d), desc="random1d"):
    start = time()
    WolfPack1d = wolf1d.Pack1D(x_min,x_max,parametric,random_points_number)
    random_points_best_value += WolfPack1d.find_prime()[1]
    stop = time()
    random_time += stop-start

result_table.contains.append([
    result_table.contains[-1][0]+1,
    "Параметрична",
    "Випадкові точки",
    "X є ["+str(x_min)+"; "+str(x_max)+"], Точок: " + str(random_points_number),
    0,
    str(round(random_time*10**6/repeat1d, 1)) + "mcs",
    str(round(random_time*10**6/repeat1d, 1)) + "mcs",
    round(random_points_best_value/repeat1d, 3),
    1.0415,
    round(abs(100 - random_points_best_value * 100 / (1.0415 * repeat1d)), 2)
])

# метод вовків, врівноважений
wolf_number = 20
wolf_iterations = 20
wolf_best_value = 0
wolf_solve_time = 0
wolf_startup_time=0

for i in tqdm(range(repeat1d), desc="wolf1d"):
    start = time()
    WolfPack1d = wolf1d.Pack1D(x_min,x_max,parametric,wolf_number)
    stop = time()
    wolf_startup_time += stop-start
    start = time()
    WolfPack1d.solve(wolf_iterations)
    wolf_best_value += WolfPack1d.find_prime()[1]
    stop = time()
    wolf_solve_time += stop-start

result_table.contains.append([
    result_table.contains[-1][0]+1,
    "Гармонічна",
    "Вовки=Ітерації",
    "X є ["+str(x_min)+"; "+str(x_max)+"], Вовків: " + str(wolf_number) + ", Ітерацій: " + str(wolf_iterations),
    str(round(wolf_startup_time * 10 ** 6 / repeat1d, 1)) + "mcs",
    str(round(wolf_solve_time * 10 ** 6 / repeat1d, 1)) + "mcs",
    str(round((wolf_startup_time + wolf_solve_time) * 10 ** 6 / repeat1d, 1)) + "mcs",
    round(wolf_best_value/repeat1d, 3),
    1.0415,
    round(abs(100 - wolf_best_value * 100 / (1.0415 * repeat1d)), 2)
])

# генетичний метод
genetic_pop_size = 15
genetic_bits = 16
genetic_iterations = 20
genetic_best_value = 0
genetic_solve_time = 0
genetic_startup_time = 0
for i in tqdm(range(repeat1d), desc="genetic1d"):
    start = time()
    geneticsolver1d = genetic1d.Genetic1D(x_min, x_max, parametric, genetic_pop_size, genetic_pop_size, genetic_bits)
    stop = time()
    genetic_startup_time += stop-start
    start = time()
    genetic_best_value += geneticsolver1d.solve(genetic_iterations)
    stop = time()
    genetic_solve_time += stop-start

result_table.contains.append([
    result_table.contains[-1][0]+1,
    #result_table.contains[-1][0]+1,
    "Параметрична",
    "Генетичний",
    "X є ["+str(x_min)+"; "+str(x_max)+"], Поп.: " + str(genetic_pop_size) + ", Біт: " + str(genetic_bits)+ ", Іт.: "+str(genetic_iterations),
    str(round(genetic_startup_time*10**6/repeat1d, 1)) + "mсs",
    str(round(genetic_solve_time*10**6/repeat1d, 1)) + "mсs",
    str(round((genetic_solve_time+genetic_startup_time)*10**6/repeat1d, 1)) + "mсs",
    round(genetic_best_value/repeat1d, 3),
    1.0415,
    round(abs(100-genetic_best_value*100/(1.0415*repeat1d)),2)
])

result_table.transpose()
print(result_table)


WolfPack1d = wolf1d.Pack1D(x_min,x_max,garmonic,wolf_number)
WolfPack1d.solve(wolf_iterations, True)

geneticsolver1d = genetic1d.Genetic1D(x_min, x_max, garmonic, genetic_pop_size, genetic_pop_size, genetic_bits)
geneticsolver1d.solve(genetic_iterations, True)

WolfPack1d = wolf1d.Pack1D(x_min,x_max,parametric,wolf_number)
WolfPack1d.solve(wolf_iterations, True)

geneticsolver1d = genetic1d.Genetic1D(x_min, x_max, parametric, genetic_pop_size, genetic_pop_size, genetic_bits)
geneticsolver1d.solve(genetic_iterations, True)