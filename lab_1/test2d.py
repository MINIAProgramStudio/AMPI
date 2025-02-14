import math
import PythonTableConsole as PTC
import numpy as np
from time import time
from tqdm import tqdm

import randompoints as rp
import wolf2d as wolf2d

result_table = PTC.PythonTableConsole([
    ["No", "Функція","Метод пошуку","Параметри","Час створення","Час пошуку", "Загальний час","Результат в середньому", "Найкращий можливий результат", "Відносна середня похибка, %"]
])

repeat1d = 10**2

x_min = -100
x_max = 100
y_min = -100
y_max = 100

def isoma(x,y):
    return -math.cos(x)*math.cos(y)*math.e**(-(x-math.pi)**2-(y-math.pi)**2)

# метод випадкових точок
random_points_number = 2*10**4
random_points_best_value = 0
random_time = 0

for i in tqdm(range(repeat1d), desc="random2d"):
    start = time()

    random_points_best_value += rp.randomsolver([[x_min, x_max],[y_min, y_max]], isoma, random_points_number, seeking_min=True)
    stop = time()
    random_time += stop-start

result_table.contains.append([
    1,
    "Ізома",
    "Випадкові точки",
    "Точок: " + str(random_points_number),
    0,
    str(round(random_time*10**3/repeat1d, 1)) + "ms",
    str(round(random_time*10**3/repeat1d, 1)) + "ms",
    round(random_points_best_value/repeat1d, 3),
    -1,
    round(abs(100-random_points_best_value*100/(-1*repeat1d)),2)
])



wolf_number = 100
wolf_iterations = 200
wolf_best_value = 0
wolf_solve_time = 0
wolf_startup_time=0

for i in tqdm(range(repeat1d), desc="wolf2d"):
    start = time()
    pack2d = wolf2d.Pack2D(x_min, x_max, y_min, y_max, isoma, wolf_number, seeking_min=True)
    stop = time()
    wolf_startup_time +=stop-start
    start = time()
    pack2d.solve(wolf_iterations)
    wolf_best_value += pack2d.find_prime()[1]
    stop = time()
    wolf_solve_time += stop-start

result_table.contains.append([
    result_table.contains[-1][0]+1,
    "Ізома",
    "Вовки",
    "Вовків: " + str(wolf_number) + ", Ітерацій: " + str(wolf_iterations),
    str(round(wolf_startup_time * 10 ** 3 / repeat1d, 1)) + "ms",
    str(round(wolf_solve_time * 10 ** 3 / repeat1d, 1)) + "ms",
    str(round((wolf_startup_time + wolf_solve_time) * 10 ** 3 / repeat1d, 1)) + "ms",
    round(wolf_best_value/repeat1d, 3),
    -1,
    round(abs(100 - wolf_best_value * 100 / (-1 * repeat1d)), 2)
])

result_table.transpose()
print(result_table)