from cheb import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor

from os import environ

from scipy.integrate import solve_ivp
from scipy.integrate import odeint

import matplotlib.pyplot as plt

from neurodiffeq import diff
from neurodiffeq.solvers import Solver1D, Solver2D
from neurodiffeq.conditions import IVP, DirichletBVP2D, DirichletBVP, BundleIVP, DoubleEndedBVP1D
from neurodiffeq.networks import FCNN, SinActv


#PARAMS
N = 1000

EPS = 1e-2

V0 = 50.0
NU = 1e-4
DELTA_V = 10.0
ZA = 1.0
ALPHA = 1.0

CA0 = 1e-2

P = 1.0

def F(y):
    return y*DELTA_V


def ode_system(Cplus, Cminus, Ca, y):
    return [V0*diff(Cplus,y)+diff(Cplus*DELTA_V + diff(Cplus,y),y),
            V0*diff(Cminus,y) + diff(-Cminus*DELTA_V + diff(Cminus,y),y),
            ALPHA*V0*diff(Ca,y) + diff(-ZA*Ca*DELTA_V + diff(Ca,y),y),
            -(Cminus+ZA*Ca-Cplus),
            (y<EPS)*(diff(Ca, y)-ZA*Ca*DELTA_V),
            (y<EPS)*(diff(Cminus, y)-Cminus*DELTA_V)
            ]


conditions = [
    DirichletBVP(
        t_0=0, u_0=P,
        t_1=1, u_1=1.0,
    ),
    IVP(t_0=1.0, u_0=(1.0-ZA*CA0)),
    IVP(t_0=1.0, u_0=CA0),
]

nets = [FCNN(actv=SinActv), FCNN(actv=SinActv), FCNN(actv=SinActv) , FCNN(actv=SinActv)]

solver = Solver1D(ode_system, conditions, t_min=0.0, t_max=1.0, nets=nets)
solver.fit(max_epochs=N)
solution = solver.get_solution()

t = np.linspace(0.0, 1.0, N)
Cplus, Cminus, Ca = solution(t, to_numpy=True)

fig, axs = plt.subplots(1, 2)

axs[0].plot(t, Cplus, label="C+")
axs[0].plot(t, Cminus, label="C-")
axs[0].plot(t, Ca, label="Ca")

axs[1].plot(t, F(t), label='F')

axs[0].legend()
axs[1].legend()
plt.show()
