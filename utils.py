import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution



def find_files(file_name):
    ''' use locate in linux to find file_name '''
    command = ['locate', file_name]
    output = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
    output = output.decode()
    search_results = output.split('\n')
    return search_results



def diff_evolution_fit(x, y, func, bounds=[(-1,1),(-1,1)], popsize=10):
    ''' wrapper to use scipy.optimize.differential_evolution() 
        on data x,y, using function func 
            x,y    : data to fit 
            func   : ref. to function used to fit x,y
            bounds : limits for each parameter of func [(min,max), (min,max), ...]
            popsize: population size
        Ex: fit a noisy parabola:
        > def f(x,a,b):
            return a*x**2 + b
        > x = linspace(-5,5,100)
        > y = 3*x**2 + 5 + randn(len(x))
        > utils.diff_evolution_fit(x, y, f, bounds=[(-5,5),(-5,7)])
        Out: array([2.99152131, 4.95353557])
    '''
    def RMSE(params, *data):
        x,y = data
        return np.sqrt(np.mean((func(x, *params)-y)**2))
    
    diffevo = differential_evolution(RMSE, bounds, args=(x,y), popsize=popsize)
    popt = diffevo.x
    return popt




