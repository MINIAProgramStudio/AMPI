import numpy as np
from PSO import PSOSolver
from matplotlib import pyplot as plt
from tqdm import tqdm
from BEE import BEEHive
from FF import  FFSolver
from FuncLim import FuncLim

def test_mean(object, iterations, tests, desc = "test_mean"):
    output = []
    for _ in tqdm(range(tests), desc = desc):
        object.reset()
        output.append([
            object.solve_stats(iterations)[2]
        ])
    return np.mean(output, axis=0)[0]

def test_time(object, iterations, tests, desc = "test_time"):
    output = []
    for _ in tqdm(range(tests), desc = desc):
        object.reset()
        output.append([
            object.solve_time(iterations)[2]
        ])
    return np.mean(output, axis=0)[0]

# functions

def rastring(pos):
    output = 10*len(pos)
    for d in range(len(pos)):
        output += pos[d]**2 - 10*np.cos(2*np.pi*pos[d])
    return output

def rosenbrok(pos):
    return (1-pos[0])**2 + 100*(pos[1] - pos[0]**2)**2

def ros_constr_1(pos):
    return (pos[0]-1)**3 - pos[1] + 1

def ros_constr_2(pos):
    return pos[0] + pos[1] - 2

def ros_constr_3(pos):
    return pos[0]**2 + pos[1]**2

lim_rosenbrok_1 = FuncLim(rosenbrok,[
    [ros_constr_1, 0, -1],
    [ros_constr_2, 0, -1]
])

lim_rosenbrok_2 = FuncLim(rosenbrok,[
    [ros_constr_3, 2, -1]
])

#optimisers
def bee_opt(pos):
    _bee = BEEHive({
        "delta": pos[0],
        "n_max": pos[1],
        "alpha": pos[2],
        "pop_size": 50,
        "l_s": int(pos[3]),
        "l_es": 10,
        "z_e": 5,
        "z_o": 2,
        "dim": 5,
        "pos_min": np.array([-5.12] * 5),
        "pos_max": np.array([5.12] * 5)
    }, rastring, True)
    result = _bee.solve(25)[0]
    return result

def ff_opt(pos):
    _ff = FFSolver({
        "b_max": pos[0],
        "gamma": pos[1],
        "alpha": pos[2],
        "pop_size": 25,
        "pos_min": np.array([-5.12] * 2),
        "pos_max": np.array([15.12] * 2),
        "dim": 2,
    }, rastring, True)
    result = _ff.solve(25)[0]
    return result
"""
pso_for_bee = PSOSolver({
"a1": 0.05,#acceleration number
"a2": 0.1,#acceleration number
"pop_size": 50,#population size
"dim": 4,#dimensions
"pos_min": np.array([0,0,0,10]),#vector of minimum positions
"pos_max": np.array([1,1,1,41]),#vector of maximum positions
"speed_min": np.array([-0.5]*3+ [5]),#vector of min speed
"speed_max": np.array([0.5]*3 + [5]),#vector of max speed
}, bee_opt, True)
#print(pso_for_bee.solve(50, True))

pso_for_ff = PSOSolver({
"a1": 0.005,#acceleration number
"a2": 0.01,#acceleration number
"pop_size": 50,#population size
"dim": 3,#dimensions
"pos_min": np.array([0,0,0]),#vector of minimum positions
"pos_max": np.array([5,1,1]),#vector of maximum positions
"speed_min": np.array([-0.1]*3),#vector of min speed
"speed_max": np.array([0.1]*3),#vector of max speed
}, ff_opt, True)
print(pso_for_ff.solve(200, True))
"""
"""
# rastring anisolve
"""
pso = PSOSolver({
"a1": 2,#acceleration number
"a2": 3,#acceleration number
"pop_size": 25,#population size
"dim": 2,#dimensions
"pos_min": np.array([-5.12]*2),#vector of minimum positions
"pos_max": np.array([15.12]*2),#vector of maximum positions
"speed_min": np.array([-0.5]*2),#vector of min speed
"speed_max": np.array([0.5]*2),#vector of max speed
}, rastring, True)

print(pso.anisolve())

bee = BEEHive({
    "delta": 0.75,
    "n_max": 0.34,
    "alpha": 0.74,
    "pop_size": 25,
    "l_s": 20,
    "l_es": 10,
    "z_e": 5,
    "z_o": 2,
    "dim": 2,
    "pos_min":  np.array([-5.12]*2),
    "pos_max": np.array([15.12]*2)
}, rastring, True)

print(bee.anisolve())

ff = FFSolver({
    "b_max": 3.22,
    "gamma": 0.009,
    "alpha": 0.981,
    "pop_size": 25,
    "pos_min": np.array([-5.12]*2),
    "pos_max": np.array([15.12]*2),
    "dim": 2,
}, rastring, True)

print(ff.anisolve())

# rastring compare

pso_y = test_mean(pso, 100, 100)
bee_y = test_mean(bee, 100, 100)
ff_y = test_mean(ff, 100, 100)


plt.plot(range(len(pso_y)),pso_y, "b", label = "PSO")
plt.plot(range(len(bee_y)), bee_y, "r", label = "BEE")
plt.plot(range(len(ff_y)), ff_y, "g", label = "FF")
plt.legend()
plt.show()

pso_x = test_time(pso, 100, 100)
bee_x = test_time(bee, 100, 100)
ff_x = test_time(ff, 100, 100)

plt.plot(pso_x,range(len(pso_x)), "b", label = "PSO")
plt.plot(bee_x,range(len(bee_x)), "r", label = "BEE")
plt.plot(ff_x,range(len(ff_x)), "g", label = "FF")
plt.legend()
plt.show()

plt.plot(pso_x,pso_y, "b", label = "PSO")
plt.plot(bee_x,bee_y, "r", label = "BEE")
plt.plot(ff_x,ff_y, "g", label = "FF")
plt.legend()
plt.show()


# rosenbrok anisolve

pso = PSOSolver({
"a1": 2,#acceleration number
"a2": 3,#acceleration number
"pop_size": 25,#population size
"dim": 2,#dimensions
"pos_min": np.array([-1.5, -0.5]),#vector of minimum positions
"pos_max": np.array([1.5, 2.5]),#vector of maximum positions
"speed_min": np.array([-0.5]*2),#vector of min speed
"speed_max": np.array([0.5]*2),#vector of max speed
}, lim_rosenbrok_1.func, True)

print(pso.anisolve())

bee = BEEHive({
    "delta": 0.75,
    "n_max": 0.34,
    "alpha": 0.74,
    "pop_size": 25,
    "l_s": 20,
    "l_es": 10,
    "z_e": 5,
    "z_o": 2,
    "dim": 2,
    "pos_min":  np.array([-1.5, -0.5]),
    "pos_max": np.array([1.5, 2.5])
}, lim_rosenbrok_1.func, True)

print(bee.anisolve())

ff = FFSolver({
    "b_max": 3.22,
    "gamma": 0.009,
    "alpha": 0.981,
    "pop_size": 25,
    "pos_min": np.array([-1.5, -0.5]),
    "pos_max": np.array([1.5, 2.5]),
    "dim": 2,
}, lim_rosenbrok_1.func, True)

print(ff.anisolve())

# rosenbrok compare

pso_y = test_mean(pso, 100, 100)
bee_y = test_mean(bee, 100, 100)
ff_y = test_mean(ff, 100, 100)


plt.plot(range(len(pso_y)),pso_y, "b", label = "PSO")
plt.plot(range(len(bee_y)), bee_y, "r", label = "BEE")
plt.plot(range(len(ff_y)), ff_y, "g", label = "FF")
plt.legend()
plt.show()

pso_x = test_time(pso, 100, 100)
bee_x = test_time(bee, 100, 100)
ff_x = test_time(ff, 100, 100)

plt.plot(pso_x,range(len(pso_x)), "b", label = "PSO")
plt.plot(bee_x,range(len(bee_x)), "r", label = "BEE")
plt.plot(ff_x,range(len(ff_x)), "g", label = "FF")
plt.legend()
plt.show()

plt.plot(pso_x,pso_y, "b", label = "PSO")
plt.plot(bee_x,bee_y, "r", label = "BEE")
plt.plot(ff_x,ff_y, "g", label = "FF")
plt.legend()
plt.show()


# rosenbrok disk anisolve

pso = PSOSolver({
"a1": 2,#acceleration number
"a2": 3,#acceleration number
"pop_size": 25,#population size
"dim": 2,#dimensions
"pos_min": np.array([-1.5, -1.5]),#vector of minimum positions
"pos_max": np.array([1.5, 1.5]),#vector of maximum positions
"speed_min": np.array([-0.5]*2),#vector of min speed
"speed_max": np.array([0.5]*2),#vector of max speed
}, lim_rosenbrok_2.func, True)

print(pso.anisolve())

bee = BEEHive({
    "delta": 0.75,
    "n_max": 0.34,
    "alpha": 0.74,
    "pop_size": 25,
    "l_s": 20,
    "l_es": 10,
    "z_e": 5,
    "z_o": 2,
    "dim": 2,
    "pos_min":  np.array([-1.5, -1.5]),
    "pos_max": np.array([1.5, 1.5])
}, lim_rosenbrok_2.func, True)

print(bee.anisolve())

ff = FFSolver({
    "b_max": 3.22,
    "gamma": 0.009,
    "alpha": 0.981,
    "pop_size": 25,
    "pos_min": np.array([-1.5, -1.5]),
    "pos_max": np.array([1.5, 1.5]),
    "dim": 2,
}, lim_rosenbrok_2.func, True)

print(ff.anisolve())

# rosenbrok compare

pso_y = test_mean(pso, 100, 100)
bee_y = test_mean(bee, 100, 100)
ff_y = test_mean(ff, 100, 100)


plt.plot(range(len(pso_y)),pso_y, "b", label = "PSO")
plt.plot(range(len(bee_y)), bee_y, "r", label = "BEE")
plt.plot(range(len(ff_y)), ff_y, "g", label = "FF")
plt.legend()
plt.show()

pso_x = test_time(pso, 100, 100)
bee_x = test_time(bee, 100, 100)
ff_x = test_time(ff, 100, 100)

plt.plot(pso_x,range(len(pso_x)), "b", label = "PSO")
plt.plot(bee_x,range(len(bee_x)), "r", label = "BEE")
plt.plot(ff_x,range(len(ff_x)), "g", label = "FF")
plt.legend()
plt.show()

plt.plot(pso_x,pso_y, "b", label = "PSO")
plt.plot(bee_x,bee_y, "r", label = "BEE")
plt.plot(ff_x,ff_y, "g", label = "FF")
plt.legend()
plt.show()


# task 5.1
def func_5_1(pos):
    return 0.7854*pos[0]*pos[1]**2 * (3.3333*pos[2]**2 + 14.9334*pos[2] - 43.0934) - 1.598*pos[0]*(pos[5]**2 + pos[6]**2) + 7.4777*(pos[5]**3 + pos[6]**3) + 0.7854*(pos[3]*pos[5]**2 + pos[4]*pos[6]**2)

def lim_5_1_1(pos):
    return 27/(pos[0]*pos[1]**2*pos[2]) - 1

def lim_5_1_2(pos):
    return 397.5/(pos[0]*pos[1]**2*pos[2]**2) - 1

def lim_5_1_3(pos):
    return 1.93*pos[3]**3/(pos[1]*pos[2]*pos[5]**4)-1

def lim_5_1_4(pos):
    return 1.93/(pos[1]*pos[2]*pos[6]**4)-1

def lim_5_1_5(pos):
    return (1/(110*pos[5]**3))*np.sqrt((745*pos[3]/(pos[1]*pos[2]))**2 + 16.9*10**6) -1

def lim_5_1_6(pos):
    return (1/(85*pos[6]**3)) * np.sqrt((745*pos[4]/(pos[1]*pos[2]))**2 + 157.5*10**6) - 1

def lim_5_1_7(pos):
    return pos[1]*pos[2]/40 -1

def lim_5_1_8(pos):
    return 5*pos[1]/pos[0] -1

def lim_5_1_9(pos):
    return pos[0]/(12*pos[1]) - 1

def lim_5_1_10(pos):
    return (1.5*pos[5]+1.9)/pos[3] - 1

def lim_5_1_11(pos):
    return (1.1*pos[6]+1.9)/pos[4] - 1

lim_5_1 = FuncLim(func_5_1, [
    [lim_5_1_1, 0, -2],
    [lim_5_1_2, 0, -2],
    [lim_5_1_3, 0, -2],
    [lim_5_1_4, 0, -2],
    [lim_5_1_5, 0, -2],
    [lim_5_1_6, 0, -2],
    [lim_5_1_7, 0, -2],
    [lim_5_1_8, 0, -2],
    [lim_5_1_9, 0, -2],
    [lim_5_1_10, 0, -2],
    [lim_5_1_11, 0, -2],
])

l51_pos_min = np.array([2.6, 0.7, 17, 7.3, 7.8, 2.9, 5])
l51_pos_max = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])

# 5_1  anisolve

pso = PSOSolver({
"a1": 0.02,#acceleration number
"a2": 0.03,#acceleration number
"pop_size": 100,#population size
"dim": 7,#dimensions
"pos_min": l51_pos_min,#vector of minimum positions
"pos_max": l51_pos_max,#vector of maximum positions
"speed_min": np.array([-0.05]*7),#vector of min speed
"speed_max": np.array([0.05]*7),#vector of max speed
}, lim_5_1.func, True)

print(pso.anisolve())

bee = BEEHive({
    "delta": 0.75,
    "n_max": 0.34,
    "alpha": 0.74,
    "pop_size": 25,
    "l_s": 20,
    "l_es": 10,
    "z_e": 5,
    "z_o": 2,
    "dim": 7,
    "pos_min":  l51_pos_min,
    "pos_max": l51_pos_max
}, lim_5_1.func, True)

print(bee.anisolve())

ff = FFSolver({
    "b_max": 3.22,
    "gamma": 0.009,
    "alpha": 0.981,
    "pop_size": 25,
    "pos_min": l51_pos_min,
    "pos_max": l51_pos_max,
    "dim": 7,
}, lim_5_1.func, True)

print(ff.anisolve())

# 5_1 compare

pso_y = test_mean(pso, 100, 25)
bee_y = test_mean(bee, 100, 25)
ff_y = test_mean(ff, 100, 25)


plt.plot(range(len(pso_y)),pso_y, "b", label = "PSO")
plt.plot(range(len(bee_y)), bee_y, "r", label = "BEE")
plt.plot(range(len(ff_y)), ff_y, "g", label = "FF")
plt.legend()
plt.show()

pso_x = test_time(pso, 100, 25)
bee_x = test_time(bee, 100, 25)
ff_x = test_time(ff, 100, 25)

plt.plot(pso_x,range(len(pso_x)), "b", label = "PSO")
plt.plot(bee_x,range(len(bee_x)), "r", label = "BEE")
plt.plot(ff_x,range(len(ff_x)), "g", label = "FF")
plt.legend()
plt.show()

plt.plot(pso_x,pso_y, "b", label = "PSO")
plt.plot(bee_x,bee_y, "r", label = "BEE")
plt.plot(ff_x,ff_y, "g", label = "FF")
plt.legend()
plt.show()


# task 5.2

def func_5_2(pos):
    return (pos[2]+2)*pos[1]*pos[0]**2

def lim_5_2_1(pos):
    return 1 - (pos[1]**3*pos[2])/(7.178*pos[0]**4)

def lim_5_2_2(pos):
    return (4*pos[1]**2 - pos[0]*pos[1])/(12.566*pos[1]*pos[2]**3-pos[0]**4) + 1/(5.108*pos[0]**2)

def lim_5_2_3(pos):
    return 1 - 140.45*pos[0]/(pos[1]**2+pos[2])

def lim_5_2_4(pos):
    return (pos[1]+pos[2])/1.5 - 1

lim_5_2 = FuncLim(func_5_2, [
    [lim_5_2_1, 0, -2],
    [lim_5_2_2, 0, -2],
    [lim_5_2_3, 0, -2],
    [lim_5_2_4, 0, -2],
])

l51_pos_min = np.array([0.005, 0.25, 2.0])
l51_pos_max = np.array([2.0, 1.3, 15])

# 5_2  anisolve

pso = PSOSolver({
"a1": 0.02,#acceleration number
"a2": 0.03,#acceleration number
"pop_size": 25,#population size
"dim": 3,#dimensions
"pos_min": l51_pos_min,#vector of minimum positions
"pos_max": l51_pos_max,#vector of maximum positions
"speed_min": np.array([-0.05]*3),#vector of min speed
"speed_max": np.array([0.05]*3),#vector of max speed
}, lim_5_2.func, True)

print(pso.anisolve())

bee = BEEHive({
    "delta": 0.75,
    "n_max": 0.34,
    "alpha": 0.74,
    "pop_size": 25,
    "l_s": 20,
    "l_es": 10,
    "z_e": 5,
    "z_o": 2,
    "dim": 3,
    "pos_min":  l51_pos_min,
    "pos_max": l51_pos_max
}, lim_5_2.func, True)

print(bee.anisolve())

ff = FFSolver({
    "b_max": 3.22,
    "gamma": 0.009,
    "alpha": 0.981,
    "pop_size": 25,
    "pos_min": l51_pos_min,
    "pos_max": l51_pos_max,
    "dim": 3,
}, lim_5_2.func, True)

print(ff.anisolve())

# 5_2 compare

pso_y = test_mean(pso, 100, 25)
bee_y = test_mean(bee, 100, 25)
ff_y = test_mean(ff, 100, 25)


plt.plot(range(len(pso_y)),pso_y, "b", label = "PSO")
plt.plot(range(len(bee_y)), bee_y, "r", label = "BEE")
plt.plot(range(len(ff_y)), ff_y, "g", label = "FF")
plt.legend()
plt.show()

pso_x = test_time(pso, 100, 25)
bee_x = test_time(bee, 100, 25)
ff_x = test_time(ff, 100, 25)

plt.plot(pso_x,range(len(pso_x)), "b", label = "PSO")
plt.plot(bee_x,range(len(bee_x)), "r", label = "BEE")
plt.plot(ff_x,range(len(ff_x)), "g", label = "FF")
plt.legend()
plt.show()

plt.plot(pso_x,pso_y, "b", label = "PSO")
plt.plot(bee_x,bee_y, "r", label = "BEE")
plt.plot(ff_x,ff_y, "g", label = "FF")
plt.legend()
plt.show()
