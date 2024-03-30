import numpy as np
import math as mth
import scipy.integrate as integrate

def myIntegrate(x, f, method = 'simp'):
    res = 0.0



    for i in range(1, len(x)):

        if method == 'trap':
            res += (f(x[i])+f(x[i-1]))/2.0 * (x[i]-x[i-1])
        elif method == 'simp':
            res += (x[i]-x[i-1])/6.0 * ( f(x[i-1]) + 4.0 * f( (x[i-1]+x[i])/2 ) + f(x[i]) )

    return res

