import numpy as np
import math as mth
import scipy.integrate as integrate
from scipy.fft import dct
from myIntegrate import *

EPS = 1e-8

def Grid(n):
    z = []
    for i in range(n):
        zi = mth.cos( (n-i-1)/(n-1) * mth.pi )
        z.append(zi)
    return np.array(z)

def GridAB(n, a, b):
    res = Grid(n)
    for i in range(n):
        res[i] = a + ( (res[i]+1.0)/2.0 ) *(b-a)
    return res

def cheb_T(x, n):
    return mth.cos(n* mth.acos(x))

def cheb_W(x):
    if x*x != 1.0 :
        return 1.0/mth.sqrt(1.0-x*x)
    else: return 0.0

def cheb_N(n):
    if n == 0:
        return mth.pi
    else:
        return mth.pi/2.0

def cheb_Coef(f, n, a=-1.0, b=1.0, eps = EPS):
    A = []
    for i in range(n):
        Ai = integrate.quad(lambda x: cheb_W(x)*  f( a + ( (x+1.0)/2.0 ) *(b-a) )  *cheb_T(x, i), -1, 1, epsabs=eps)[0]
        if abs(Ai) < eps: Ai = 0.0
        A.append(Ai/cheb_N(i))
    return A

def cheb_Coef2(f, n, a = -1.0, b=1.0):
    z = Grid(n)
    f2 = lambda x: f( a + ( (-x+1.0)/2.0 ) *(b-a) )
    y = f2(z)
    A = np.array(dct(y, 1))
    inv = 1.0/(n)
    A = A*inv
    A[0] = A[0]/2.0
    A[-1] = A[-1]/2.0
    return A


def cheb_Diff(coefs, n, a=-1.0, b=1.0):
    res = list(np.zeros(n))
    for p in range(1, n, 2):
        res[0] += p*coefs[p]
    for m in range(1, n-1):
        s = 0.0
        for p in range(m+1, n, 2):
            s += p*coefs[p]
        res[m] = 2.0 * s
    return np.array(res) * (b-a)/2.0

def cheb_DiffP(coefs, n, p):
    res = coefs
    for i in range(p):
        res = cheb_Diff(res, n)
    return res

def cheb_Fx(coefs, n):
    res = []
    z = Grid(n)
    #print(z)
    for i in range(n):
        fi = 0.0
        for j in range(n):
            fi += cheb_T(z[i], j) * coefs[j]
        res.append(fi)
    return np.array(res)

def Discrepancy(f, coefs, n, a=-1.0, b=1.0):
    def f2(x, n):
        res = 0.0
        for j in range(n):
            res += cheb_T(x, j) * coefs[j]
        return res
    return integrate.quad(lambda x: np.abs( f(x) - f2(x, n) )**2, a, b)[0]

def cheb_Mul(coefs1, coefs2, n):
    res = np.zeros(n)
    for i in range(1, n):
        res[0] += 0.5 * coefs1[i]*coefs2[i]
    res[0] += coefs1[0]*coefs2[0]

    for i in range(1, n):
        for j in range(i, n):
            res[i] += 0.5*( coefs1[j] * coefs2[j-i] + coefs2[j]*coefs1[j-i] )
        for j in range(i):
            res[i] += 0.5 * coefs1[j] * coefs2[i-j]
    return res

def cheb_Int(coefs, n):
    res = np.zeros(n)
    for i in range(2, n-1):
        res[i] = 0.5 * (coefs[i-1]-coefs[i+1])/(i)
    res[n-1] = 0.5 * coefs[n-2]/(n-1.0)
    res[1] = coefs[0]-0.5*coefs[2]
    res[0] = 0.0

    p1 = 0.0
    for j in range(n):
        p1 += cheb_T(-1.0, j) * res[j]

    p2 = 0.0
    for j in range(n):
        p2 += cheb_T(1.0, j) * res[j]

    return res, p1, p2, p2-p1



'''def f1(x):
    return x*x/2.0'''

#print(integrate.quad(f1, -1, 1)[0])
#print(cheb_Int(cheb_Coef2(f1, 512, -1, 1), 512)[-1])
