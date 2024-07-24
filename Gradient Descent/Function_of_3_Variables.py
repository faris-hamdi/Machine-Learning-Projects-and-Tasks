import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


def gradient_descent(fderiv, inital_start, step_size = 0.001, precision = 0.00001, max_iter = 1000):
    cur_start = np.array(inital_start)
    last_start = cur_start + 100 * precision    # something different
    start_list = []

    iter = 0
    while norm(cur_start - last_start) > precision and iter < max_iter:
        print(cur_start)
        last_start = cur_start.copy()     # must copy

        gradient = fderiv(cur_start)
        cur_start -= gradient * step_size   # move in opposite direction

        start_list.append(cur_start.copy())
        iter += 1

    return cur_start

def func(x,y,z):
    return np.sin(x) + np.cos(y) + np.sin(z)

def fderiv_dx(x,y,z):
    return np.cos(x)
    
def fderiv_dy(x,y,z):
    return - np.sin(y)
    
def fderiv_dz(x,y,z):
    return np.cos(z)

def fderiv(state):
    x, y, z = state[0], state[1], state[2]
    return np.array([fderiv_dx(x,y,z), fderiv_dy(x,y,z), fderiv_dz(x,y,z)])

start_point= [1., 2., 3.5]

optima = gradient_descent(fderiv, start_point)
    
