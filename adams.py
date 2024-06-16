import numpy as np
from cheb import *

#import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def adams_bashforth(f, y0, steps, delta_x):
    n = len(y0)
    y_array = [y0]
    x_array = [0.0]


    # Начальный шаг методом Рунге-Кутты 4-го порядка
    for i in range(3):
        print('STEP NUMBER', i)
        y = np.zeros(n)
        #print(f(x_array[i], y_array[i]))
        print("k1")
        k1 = delta_x * f(x_array[i], y_array[i])
        print("k1=", k1)
        print("k2")
        k2 = delta_x * f(x_array[i] + delta_x / 2.0, y_array[i] + k1 / 2.0 )
        print("k2=", k2)
        print("k3")
        k3 = delta_x * f(x_array[i] + delta_x / 2.0, y_array[i] + k2 / 2.0 )
        print("k3=", k3)
        print("k4")
        k4 = delta_x * f(x_array[i] + delta_x, y_array[i] + k3 )
        print("k4=", k4)
        #print(k4)
        y = y_array[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        y_array.append(y)
        #print(y)
        x_array.append(x_array[i] + delta_x)

    # Шаги методом Адамса-Башфорта
    for i in range(3, steps):
        print('STEP NUMBER', i)
        y = np.zeros(n)
        y = y_array[i] + delta_x * (55 * f(x_array[i], y_array[i]) - 59 * f(x_array[i-1], y_array[i-1]) + 37 * f(x_array[i-2], y_array[i-2]) - 9 * f(x_array[i-3], y_array[i-3])) / 24
        y_array.append(y)
        x_array.append(x_array[i] + delta_x)

    #print(y_array)

    return np.array(x_array), y_array

def adams_bashforth_moulton(f, y0, steps, delta_x):
    n = len(y0)
    y_array = [y0]
    x_array = [0.0]


    for i in range(1, steps):
        x_array.append(x_array[i-1] + delta_x)
        # Adams-Bashforth predictor
        y_pred = y_array[i-1] + delta_x/2 * (3*f(x_array[i-1], y_array[i-1]) - f(x_array[i-2], y_array[i-2]))

        # Adams-Moulton corrector
        y_array.append(y_array[i-1] + delta_x/2 * (f(x_array[i], y_pred) + f(x_array[i-1], y_array[i-1])))


    #print(y_array)

    return np.array(x_array), y_array


def rk4_process(k, delta_x, i, x_array, y_array, f):
    k1 = delta_x * f(x_array[i], y_array[i])[k]
    k2 = delta_x * f(x_array[i] + delta_x / 2, y_array[i] + k1 / 2)[k]
    k3 = delta_x * f(x_array[i] + delta_x / 2, y_array[i] + k2 / 2)[k]
    k4 = delta_x * f(x_array[i] + delta_x, y_array[i] + k3)[k]
    return y_array[i][k] + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def ab_process(k, delta_x, i, x_array, y_array, f):
    return y_array[i][k] + delta_x * (55 * f(x_array[i], y_array[i])[k] - 59 * f(x_array[i-1], y_array[i-1])[k] + 37 * f(x_array[i-2], y_array[i-2])[k] - 9 * f(x_array[i-3], y_array[i-3])[k]) / 24

def adams_bashforth_multiproccess(f, y0, steps, delta_x):
    n = len(y0)
    y_array = [y0]
    x_array = [0.0, delta_x]


    # Начальный шаг методом Рунге-Кутты 4-го порядка
    for i in range(3):
        #print(i)
        y = range(n)

        with ProcessPoolExecutor() as executor:
            executor.map(lambda k: rk4_process(k, delta_x, i, x_array, y_array, f), y)
        y_array.append(y)
        x_array.append(x_array[i] + delta_x)

    # Шаги методом Адамса-Башфорта
    for i in range(3, steps):
        #print(i)
        y = range(n)
        with ProcessPoolExecutor() as executor:
            executor.map(lambda k: ab_process(k, delta_x, i, x_array, y_array, f), y)
        y_array.append(y)
        x_array.append(x_array[i] + delta_x)

    #print(y_array)

    return x_array, y_array





'''N = 5
h = 1e-1
def f(x, y):
    return np.array([y[1], -y[0], x*x, x*x])


x_arr, y_arr = adams_bashforth_classic(f, np.array([1, 1, 0, 0]), 1000, h)



#print(y_arr)

y_arr = np.matrix(y_arr).T

print(np.array(y_arr[0])[0])
print(x_arr)
plt.plot(x_arr, np.array(y_arr[0])[0])
plt.plot(x_arr, np.array(y_arr[1])[0])
#plt.plot(x_arr, np.exp(x_arr))
plt.show()'''
