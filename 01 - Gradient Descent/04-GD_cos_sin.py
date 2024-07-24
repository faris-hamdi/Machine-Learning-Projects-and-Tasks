import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from Gradient_Descent import gradient_descent

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
    
