import numpy as np
import math as mth
import scipy.integrate as integrate
from myIntegrate import *

EPS = 1e-5
ND = 5
N = 1000

def cheb_T(x, n):
    return mth.cos(n* mth.acos(x))

def cheb_W(x):
    if x*x != 1.0 :
        return 1.0/mth.sqrt(1-x*x)
    else: return 0.0

def cheb_N(n):
    if n == 0:
        return mth.pi
    else:
        return mth.pi/2.0

'''def cheb_SP(n):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return sp.expand(2*x*cheb_SP(n-1) - cheb_SP(n-2))'''

def cheb_Coef(x, f, n, eps = EPS):
    A = []
    for i in range(n):
        Ai = integrate.quad(lambda x: cheb_W(x)*f(x)*cheb_T(x, i), -1, 1, epsabs=eps)[0]
        #print('bint', i,  Ai)
        if abs(Ai) < eps: Ai = 0.0
        A.append(Ai/cheb_N(i))
    return A

def cheb_Coef2(x, f, n):
    A = []
    for i in range(n):
        Ai = myIntegrate(x, lambda x: cheb_W(x)*f(x)*cheb_T(x, i), method='simp')
        #print('mint', i, Ai)
        A.append(Ai/cheb_N(i))

    return A


def cheb_Diff(coefs, n):
    res = list(np.zeros(n))
    for p in range(1, n, 2):
        res[0] += p*coefs[p]
    for m in range(1, n-1):
        s = 0.0
        for p in range(m+1, n, 2):
            s += p*coefs[p]
        res[m] = 2.0 * s
    return res

def cheb_Fx(coefs, x, n):
    res = 0.0
    for i in range(n):
        res += cheb_T(x, i) * coefs[i]
    return res


z = []

for i in range(N):
    z.append(mth.cos( (N-i-1)/(N-1) * mth.pi ))
z = np.array(z)

def f(x):
    return x*x

def df(x):
    return 2*x

fc = cheb_Coef2(z, f, N)
fc2 = cheb_Coef(z, f, N)

#print(z)

print(fc[0])
print(fc2[0])

#print(myIntegrate(z, lambda x: cheb_T(x, 10)))
#print(integrate.quad(lambda x: cheb_T(x, 10), -1, 1)[0])


#dfc2 = cheb_Coef2(z, df, N)

#dfc = cheb_Diff(fc, N)



#print(dfc2)
#print(dfc)
