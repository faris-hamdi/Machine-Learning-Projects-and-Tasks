import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm

def gradient_descent(fderiv, initial_start, precision=0.00001, step_size=0.001, max_iter=10000):

    cur_point = np.array(initial_start)
    iteration= 0
    points= []
    last_point= np.array([0.0, 0.0])
    last_point= np.array(last_point)

    while norm(last_point-cur_point) > precision and iteration < max_iter:
        iteration+=1
        last_point = cur_point.copy()
        x ,y = cur_point
        gradient= np.array(fderiv(x,y))
        cur_point-= gradient*step_size
        points.append(cur_point.copy())

    return cur_point, points


def func(x,y):
    return 3 * (x + 2) ** 2 + (y - 1) ** 2

def dx_func(x,y):
    return 6 * (x + 2)

def dy_func(x,y):
    return 2 * (y - 1)

def fderiv(x,y):
    return np.array([dx_func(x,y), dy_func(x,y)])

start_point = [-5.0, 2.5]

optima_point, history_points = gradient_descent(fderiv, start_point)

print(optima_point)
















