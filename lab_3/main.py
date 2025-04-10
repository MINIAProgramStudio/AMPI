from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

import CC
from CC import CC
from BatAlgorythm import BatSwarm
from PSO import PSOSolver
from FuncLim import FuncLim


def test_mean(object, iterations, tests, desc="test_mean"):
    output = []
    for _ in tqdm(range(tests), desc=desc):
        object.reset()
        output.append([
            object.solve_stats(iterations)[2]
        ])

    return np.mean(output, axis=0)[0]


def rastring(pos):
    output = 10 * len(pos)
    for d in range(len(pos)):
        output += pos[d] ** 2 - 10 * np.cos(2 * np.pi * pos[d])
    return output


def cc_optimiser(pos):
    _cc = CC({
        "p_detect": pos[0],
        "delta": pos[1],

        "pop_size": 25,
        "dim": 20,
        "pos_min": np.array([-5.12] * 20),
        "pos_max": np.array([5.12] * 20)
    }, rastring, True)
    return _cc.solve(1000, False)[0]


def bs_optimiser(pos):
    _bs = BatSwarm({
        "r0": pos[0],
        "A0": pos[1],
        "alpha": pos[2],
        "delta": pos[3],
        "gamma": pos[4],
        "freq_min": pos[5],
        "freq_max": pos[6],

        "pop_size": 50,
        "dim": 20,
        "pos_min": np.array([-5.12] * 20),
        "pos_max": np.array([5.12] * 20)
    }, rastring, True)
    return _bs.solve(100, False)[0]


def bs_min_freq_l_max_freq(pos):
    return pos[6] - pos[5]


bs_optimiser_2 = FuncLim(bs_optimiser, [
    [bs_min_freq_l_max_freq, 0, 1]
])
"""
pso_for_cc = PSOSolver({
"a1": 0.005,#acceleration number
"a2": 0.01,#acceleration number
"pop_size": 10,#population size
"dim": 2,#dimensions
"pos_min": np.array([0,0]),#vector of minimum positions
"pos_max": np.array([1,1]),#vector of maximum positions
"speed_min": np.array([-0.5]*2),#vector of min speed
"speed_max": np.array([0.5]*2),#vector of max speed
}, cc_optimiser, True)
print(pso_for_cc.solve(100, True))


pso_for_bs = PSOSolver({
"a1": 0.005,#acceleration number
"a2": 0.01,#acceleration number
"pop_size": 10,#population size
"dim": 7,#dimensions
"pos_min": np.array([0,1,0,0,0,0,0]),#vector of minimum positions
"pos_max": np.array([1,2,1,1,1,100,100]),#vector of maximum positions
"speed_min": np.array([-0.5]*7),#vector of min speed
"speed_max": np.array([0.5]*7),#vector of max speed
}, bs_optimiser_2.func, True)
print(pso_for_bs.solve(100, True))
"""

"""
cc_rastring = CC({
    "p_detect": 0.02,
    "delta": 0.0025,
    "pop_size": 10,
    "dim": 20,
    "pos_min": np.array([-5.12] * 20),
    "pos_max": np.array([5.12] * 20)
}, rastring, True)



cc_y = test_mean(cc_rastring, 10000, 20)
plt.plot(range(len(cc_y)),cc_y, "b", label = "CC")
plt.legend()
plt.show()

bs_rastring = BatSwarm({
    "r0": 0.5,
    "A0": 1.5,
    "alpha": 0.5,
    "delta": 0.5,
    "gamma": 0.5,
    "freq_min": 1,
    "freq_max": 2,

    "pop_size": 10,
    "dim": 20,
    "pos_min": np.array([-5.12] * 20),
    "pos_max": np.array([5.12] * 20)
}, rastring, True)

bs_y = test_mean(bs_rastring, 1000, 20)
plt.plot(range(len(bs_y[10:])),bs_y[10:], "b", label = "BS")
plt.legend()
plt.show()

bs_rastring = BatSwarm({
    "r0": 0.937,
    "A0": 1.169,
    "alpha": 0.95,
    "delta": 0.62,
    "gamma": 0.065,
    "freq_min": 2.37,
    "freq_max": 89.7,

    "pop_size": 10,
    "dim": 20,
    "pos_min": np.array([-5.12] * 20),
    "pos_max": np.array([5.12] * 20)
}, rastring, True)

bs_y = test_mean(bs_rastring, 1000, 20)
plt.plot(range(len(bs_y[10:])),bs_y[10:], "b", label = "BS")
plt.legend()
plt.show()



def plot_511(tN, aN):
    ts = np.linspace(1, 3, tN)
    al = np.linspace(-3, 3, aN)
    h = ts[1] - ts[0]
    for a in al:
        z = [1]
        x = [a]
        y = [1]
        for i in range(tN):
            dx = z[-1]
            dy = y[-1] ** 4 + x[-1] ** 3 - 3 * np.sin(ts[i] * z[-1])
            dz = (x[-1] ** 2) * (ts[i] ** 2) - (dy ** 2) * np.cos(z[-1])

            x.append(x[-1] + dx * h)
            y.append(y[-1] + dy * h)
            z.append(z[-1] + dz * h)
            if abs(x[-1]) + abs(y[-1]) > 10000:
                break
            plt.plot(x, y)
    plt.show()


#plot_511(100, 1000)


def plot_target_511(tN, aN):
    ts = np.linspace(1, 3, tN)
    al = np.linspace(-3, 3, aN)
    h = ts[1] - ts[0]
    f_1 = []
    f_2 = []
    for a in al:
        z = [1]
        x = [a]
        y = [1]
        all_good = True
        for i in range(tN):
            dx = z[-1]
            dy = y[-1] ** 4 + x[-1] ** 3 - 3 * np.sin(ts[i] * z[-1])
            dz = (x[-1] ** 2) * (ts[i] ** 2) - (dy ** 2) * np.cos(z[-1])

            x.append(x[-1] + dx * h)
            y.append(y[-1] + dy * h)
            z.append(z[-1] + dz * h)
            if abs(x[-1]) > 10 ** 9:
                f_1.append(abs(x[-1] - 3))
                f_2.append((x[-1] - 3)**2)
                all_good = False
                break
        if all_good:
            f_1.append(abs(x[-1] - 3))
            f_2.append((x[-1] - 3) ** 2)
    plt.plot(al, f_1, label = "abs")
    plt.plot(al, f_2, label="sq")
    plt.yscale("log")
    plt.legend()
    plt.show()


plot_target_511(1000, 1000)


def alpha_target_511_abs(alpha):
    ts = np.linspace(1, 3, 1000)
    h = ts[1] - ts[0]
    z = [1]
    x = [alpha]
    y = [1]
    for i in range(1000):
        dx = z[-1]
        dy = y[-1] ** 4 + x[-1] ** 3 - 3 * np.sin(ts[i] * z[-1])
        dz = (x[-1] ** 2) * (ts[i] ** 2) - (dy ** 2) * np.cos(z[-1])

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) + abs(y[-1]) > 10000:
            return float("inf")
    return abs(x[-1] - 3)

def alpha_target_511_sq(alpha):
    ts = np.linspace(1, 3, 1000)
    h = ts[1] - ts[0]
    z = [1]
    x = [alpha]
    y = [1]
    for i in range(1000):
        dx = z[-1]
        dy = y[-1] ** 4 + x[-1] ** 3 - 3 * np.sin(ts[i] * z[-1])
        dz = (x[-1] ** 2) * (ts[i] ** 2) - (dy ** 2) * np.cos(z[-1])

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) + abs(y[-1]) > 10000:
            return float("inf")
    return (x[-1] - 3)**2

def plot_alpha_target_511(tN, alpha):
    ts = np.linspace(1, 3, tN)
    h = ts[1] - ts[0]
    f = []
    z = [1]
    x = [alpha]
    y = [1]
    for i in range(tN):
        dx = z[-1]
        dy = y[-1] ** 4 + x[-1] ** 3 - 3 * np.sin(ts[i] * z[-1])
        dz = (x[-1] ** 2) * (ts[i] ** 2) - (dy ** 2) * np.cos(z[-1])

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) > 10 ** 9:
                break
    plt.plot(x, y)
    plt.show()
    plt.plot(ts, x[:-1])
    plt.show()

cc_511_abs = CC({
    "p_detect": 0.02,
    "delta": 0.25,
    "pop_size": 10,
    "dim": 1,
    "pos_min": [-3],
    "pos_max": [3]
}, alpha_target_511_abs, True)

bat_511_abs = BatSwarm({
    "r0": 0.937,
    "A0": 1.169,
    "alpha": 0.95,
    "delta": 0.62,
    "gamma": 0.065,
    "freq_min": 2.37,
    "freq_max": 89.7,

    "pop_size": 5,
    "dim": 1,
    "pos_min": [-3],
    "pos_max": [3]
}, alpha_target_511_abs, True)

cc_511_sq = CC({
    "p_detect": 0.02,
    "delta": 0.25,
    "pop_size": 10,
    "dim": 1,
    "pos_min": [-3],
    "pos_max": [3]
}, alpha_target_511_sq, True)

bat_511_sq = BatSwarm({
    "r0": 0.937,
    "A0": 1.169,
    "alpha": 0.95,
    "delta": 0.62,
    "gamma": 0.065,
    "freq_min": 2.37,
    "freq_max": 89.7,

    "pop_size": 5,
    "dim": 1,
    "pos_min": [-3],
    "pos_max": [3]
}, alpha_target_511_sq, True)



y_cc_abs = test_mean(cc_511_abs, 100, 5, "511_cc_abs")
y_bat_abs = test_mean(bat_511_abs, 100, 5, "511_bat_abs")
y_cc_sq = np.sqrt(test_mean(cc_511_sq, 100, 5, "511_cc_sq"))
y_bat_sq = np.sqrt(test_mean(bat_511_sq, 100, 5, "511_bat_sq"))

plt.plot(range(len(y_cc_abs)),y_cc_abs, label="CC_abs")
plt.plot(range(len(y_bat_abs)),y_bat_abs, label="Bat_abs")
plt.plot(range(len(y_cc_sq)),y_cc_sq, label="CC_sq")
plt.plot(range(len(y_bat_sq)),y_bat_sq, label="Bat_sq")
plt.legend()
plt.show()

def plot_512(tN, aN):
    ts = np.linspace(1, 3, tN)
    al = np.linspace(-10, 10, aN)
    h = ts[1] - ts[0]
    for a in al:
        z = [a]
        x = [2]
        y = [2]
        for i in range(tN):
            dx = 5*x[-1]**2-25*ts[i]**2+(z[-1]**2)*np.cos(2*ts[i]+x[-1])
            dy = z[-1]
            dz = 4/(1+(y[-1]**2)*(x[-1]**2))+4*np.sin(ts[i]*dx)

            x.append(x[-1] + dx * h)
            y.append(y[-1] + dy * h)
            z.append(z[-1] + dz * h)
            if abs(x[-1]) + abs(y[-1]) > 10000:
                break
            plt.plot(x, y)
    plt.show()
plot_512(100,100)

def plot_target_512(tN, aN):
    ts = np.linspace(1, 3, tN)
    al = np.linspace(-10, 10, aN)
    h = ts[1] - ts[0]
    f_1 = []
    f_2 = []
    for a in al:
        z = [a]
        x = [2]
        y = [2]
        all_good = True
        for i in range(tN):
            dx = 5 * x[-1] ** 2 - 25 * ts[i] ** 2 + (z[-1] ** 2) * np.cos(2 * ts[i] + x[-1])
            dy = z[-1]
            dz = 4 / (1 + (y[-1] ** 2) * (x[-1] ** 2)) + 4 * np.sin(ts[i] * dx)

            x.append(x[-1] + dx * h)
            y.append(y[-1] + dy * h)
            z.append(z[-1] + dz * h)
            if abs(x[-1]) > 10 ** 9:
                f_1.append(abs(z[-1]))
                f_2.append(z[-1]**2)
                all_good = False
                break
        if all_good:
            f_1.append(abs(z[-1]))
            f_2.append(z[-1] ** 2)
    plt.plot(al, f_1, label="abs")
    plt.plot(al, f_2, label="sq")
    plt.yscale("log")
    plt.legend()
    plt.show()


plot_target_512(1000, 10000)


def alpha_target_512_abs(alpha):
    ts = np.linspace(1, 3, 1000)
    h = ts[1] - ts[0]
    z = [alpha]
    x = [2]
    y = [2]
    for i in range(1000):
        dx = 5 * x[-1] ** 2 - 25 * ts[i] ** 2 + (z[-1] ** 2) * np.cos(2 * ts[i] + x[-1])
        dy = z[-1]
        dz = 4 / (1 + (y[-1] ** 2) * (x[-1] ** 2)) + 4 * np.sin(ts[i] * dx)

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) + abs(y[-1]) > 10000:
            return float("inf")
    return abs(z[-1])

def alpha_target_512_sq(alpha):
    ts = np.linspace(1, 3, 1000)
    h = ts[1] - ts[0]
    z = [alpha]
    x = [2]
    y = [2]
    for i in range(1000):
        dx = 5 * x[-1] ** 2 - 25 * ts[i] ** 2 + (z[-1] ** 2) * np.cos(2 * ts[i] + x[-1])
        dy = z[-1]
        dz = 4 / (1 + (y[-1] ** 2) * (x[-1] ** 2)) + 4 * np.sin(ts[i] * dx)

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) + abs(y[-1]) > 10000:
            return float("inf")
    return (z[-1])**2

def plot_alpha_target_512(tN, alpha):
    ts = np.linspace(1, 3, tN)
    h = ts[1] - ts[0]
    f = []
    z = [alpha]
    x = [2]
    y = [2]
    for i in range(tN):
        dx = 5 * x[-1] ** 2 - 25 * ts[i] ** 2 + (z[-1] ** 2) * np.cos(2 * ts[i] + x[-1])
        dy = z[-1]
        dz = 4 / (1 + (y[-1] ** 2) * (x[-1] ** 2)) + 4 * np.sin(ts[i] * dx)

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) > 10 ** 9:
                break
    plt.plot(x, y)
    plt.show()
    plt.plot(ts, x[:-1])
    plt.show()

cc_512_abs = CC({
    "p_detect": 0.02,
    "delta": 0.25,
    "pop_size": 10,
    "dim": 1,
    "pos_min": [-10],
    "pos_max": [10]
}, alpha_target_512_abs, True)

bat_512_abs = BatSwarm({
    "r0": 0.937,
    "A0": 1.169,
    "alpha": 0.95,
    "delta": 0.62,
    "gamma": 0.065,
    "freq_min": 2.37,
    "freq_max": 89.7,

    "pop_size": 5,
    "dim": 1,
    "pos_min": [-10],
    "pos_max": [10]
}, alpha_target_512_abs, True)

cc_512_sq = CC({
    "p_detect": 0.02,
    "delta": 0.25,
    "pop_size": 10,
    "dim": 1,
    "pos_min": [-10],
    "pos_max": [10]
}, alpha_target_512_sq, True)

bat_512_sq = BatSwarm({
    "r0": 0.937,
    "A0": 1.169,
    "alpha": 0.95,
    "delta": 0.62,
    "gamma": 0.065,
    "freq_min": 2.37,
    "freq_max": 89.7,

    "pop_size": 5,
    "dim": 1,
    "pos_min": [-10],
    "pos_max": [10]
}, alpha_target_512_sq, True)


y_cc_abs = test_mean(cc_512_abs, 100, 5, "511_cc_abs")
y_bat_abs = test_mean(bat_512_abs, 100, 5, "511_bat_abs")
y_cc_sq = np.sqrt(test_mean(cc_512_sq, 100, 5, "511_cc_sq"))
y_bat_sq = np.sqrt(test_mean(bat_512_sq, 100, 5, "511_bat_sq"))

plt.plot(range(len(y_cc_abs)),y_cc_abs, label="CC_abs")
plt.plot(range(len(y_bat_abs)),y_bat_abs, label="Bat_abs")
plt.plot(range(len(y_cc_sq)),y_cc_sq, label="CC_sq")
plt.plot(range(len(y_bat_sq)),y_bat_sq, label="Bat_sq")
plt.legend()
plt.show()
"""

def plot_513(tN, aN):
    ts = np.linspace(1, 3, tN)
    al = np.linspace(-10, 10, aN)
    h = ts[1] - ts[0]
    for a in al:
        z = [a]
        x = [2]
        y = [1]
        for i in range(tN):
            dx = 2*x[-1]**2-25*ts[i]**2-np.sin(x[-1]*y[-1]*ts[i])
            dz = 1-4*np.cos(dx*ts[i])
            dy = z[-1]

            x.append(x[-1] + dx * h)
            y.append(y[-1] + dy * h)
            z.append(z[-1] + dz * h)
            if abs(x[-1]) + abs(y[-1]) > 10000:
                break
            plt.plot(x, y)
    plt.show()
plot_513(1000,100)

def plot_target_513(tN, aN):
    ts = np.linspace(1, 3, tN)
    al = np.linspace(-10, 10, aN)
    h = ts[1] - ts[0]
    f_1 = []
    f_2 = []
    for a in al:
        z = [a]
        x = [2]
        y = [1]
        all_good = True
        for i in range(tN):
            dx = 2 * x[-1] ** 2 - 25 * ts[i] ** 2 - np.sin(x[-1] * y[-1] * ts[i])
            dz = 1 - 4 * np.cos(dx * ts[i])
            dy = z[-1]

            x.append(x[-1] + dx * h)
            y.append(y[-1] + dy * h)
            z.append(z[-1] + dz * h)
            if abs(x[-1]) > 10 ** 9:
                f_1.append(abs(z[-1]+1))
                f_2.append((z[-1] + 1)**2)
                all_good = False
                break
        if all_good:
            f_1.append(abs(z[-1]+1))
            f_2.append((z[-1] + 1) ** 2)
    plt.plot(al, f_1, label = "abs")
    plt.plot(al, f_2, label="sq")
    plt.yscale("log")
    plt.legend()
    plt.show()


plot_target_513(1000, 1000)


def alpha_target_513_abs(alpha):
    ts = np.linspace(1, 3, 1000)
    h = ts[1] - ts[0]
    z = [alpha]
    x = [2]
    y = [1]
    for i in range(1000):
        dx = 2 * x[-1] ** 2 - 25 * ts[i] ** 2 - np.sin(x[-1] * y[-1] * ts[i])
        dz = 1 - 4 * np.cos(dx * ts[i])
        dy = z[-1]

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) + abs(y[-1]) > 10000:
            return float("inf")
    return abs(z[-1]+1)

def alpha_target_513_sq(alpha):
    ts = np.linspace(1, 3, 1000)
    h = ts[1] - ts[0]
    z = [alpha]
    x = [2]
    y = [1]
    for i in range(1000):
        dx = 2 * x[-1] ** 2 - 25 * ts[i] ** 2 - np.sin(x[-1] * y[-1] * ts[i])
        dz = 1 - 4 * np.cos(dx * ts[i])
        dy = z[-1]

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) + abs(y[-1]) > 10000:
            return float("inf")
    return (z[-1]+1)**2

def plot_alpha_target_513(tN, alpha):
    ts = np.linspace(1, 3, tN)
    h = ts[1] - ts[0]
    f = []
    z = [alpha]
    x = [2]
    y = [1]
    for i in range(tN):
        dx = 2 * x[-1] ** 2 - 25 * ts[i] ** 2 - np.sin(x[-1] * y[-1] * ts[i])
        dz = 1 - 4 * np.cos(dx * ts[i])
        dy = z[-1]

        x.append(x[-1] + dx * h)
        y.append(y[-1] + dy * h)
        z.append(z[-1] + dz * h)
        if abs(x[-1]) > 10 ** 9:
                break
    plt.plot(x, y)
    plt.show()
    plt.plot(ts, x[:-1])
    plt.show()

cc_513_abs = CC({
    "p_detect": 0.02,
    "delta": 0.25,
    "pop_size": 10,
    "dim": 1,
    "pos_min": [-10],
    "pos_max": [10]
}, alpha_target_513_abs, True)

bat_513_abs = BatSwarm({
    "r0": 0.937,
    "A0": 1.169,
    "alpha": 0.95,
    "delta": 0.62,
    "gamma": 0.065,
    "freq_min": 2.37,
    "freq_max": 89.7,

    "pop_size": 5,
    "dim": 1,
    "pos_min": [-10],
    "pos_max": [10]
}, alpha_target_513_abs, True)

cc_513_sq = CC({
    "p_detect": 0.02,
    "delta": 0.25,
    "pop_size": 10,
    "dim": 1,
    "pos_min": [-10],
    "pos_max": [10]
}, alpha_target_513_sq, True)

bat_513_sq = BatSwarm({
    "r0": 0.937,
    "A0": 1.169,
    "alpha": 0.95,
    "delta": 0.62,
    "gamma": 0.065,
    "freq_min": 2.37,
    "freq_max": 89.7,

    "pop_size": 5,
    "dim": 1,
    "pos_min": [-10],
    "pos_max": [10]
}, alpha_target_513_sq, True)



y_cc_abs = test_mean(cc_513_abs, 100, 5, "513_cc_abs")
y_bat_abs = test_mean(bat_513_abs, 100, 5, "513_bat_abs")
y_cc_sq = test_mean(cc_513_sq, 100, 5, "513_cc_sq")
y_bat_sq = test_mean(bat_513_sq, 100, 5, "513_bat_sq")
plt.plot(range(len(y_cc_abs)),y_cc_abs, label = "cc_abs")
plt.plot(range(len(y_bat_abs)),y_bat_abs, label = "bat_abs")
plt.plot(range(len(y_cc_sq)),y_cc_sq, label = "cc_sq")
plt.plot(range(len(y_bat_sq)),y_bat_sq, label = "bat_sq")
plt.legend()
plt.show()