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

    #print(res)
    return res

'''def f(x):
    return mth.cos(x)

N = 10

x = np.linspace(-1, 1, N)
z = []
for i in range(N):
    z.append(mth.cos( (N-i-1)/(N-1) * mth.pi ))
z = np.array(z)

print(myIntegrate(z, f))
print(integrate.quad(f, -1, 1)[0])'''
