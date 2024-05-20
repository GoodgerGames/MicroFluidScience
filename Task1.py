from cheb import *
from adams import adams_bashforth, adams_bashforth_multiproccess, adams_bashforth_moulton

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt


from scipy.integrate import solve_ivp
from scipy.integrate import odeint

N1 = 8

N = N1-2

EPS = 1e-2

L = 1.0

delta_t = 1e-3

Steps = 100

C0plus = 0.5
C0minus = 0.5

K0 = np.zeros(N1)
R0 = np.zeros(N1)


P = 1.0
ZA = -1.0
A_DA = 2.0
CA0 = 1e-4
DELTA_V = 50.0
V0 = 0.0

K0[:] = 2.0 - (ZA-1.0)*CA0
#print(K0)

z = Grid(N1)

y = 0.5 * L * (z+1.0)

yF = np.zeros(N1*2-4)
yp = np.zeros(N1*2-4)

Rcoefs = cheb_Coef3(R0, N1)
Kcoefs = cheb_Coef3(K0, N1)
yF[:N] = Kcoefs[:N]
yF[N:] = Rcoefs[:N]
neqn = 2*N
cur = 0.0

Carray = []

'''res_n1 = []
for i in range(N):
    Ti = np.zeros(N)
    Ti[i] = 1.0
    dTi = cheb_Diff(Ti, N)
    res_n1.append( cheb_Fx(dTi, N)[0] )
res_n1 = np.array(res_n1)

res_n2 = []
for i in range(N):
    Ti = np.zeros(N)
    Ti[i] = 1.0
    dTi = cheb_Diff(Ti, N)
    res_n2.append( cheb_Fx(dTi, N)[N-1] )
res_n2 = np.array(res_n2)'''


def FCN(t, XX, YP = [], Carray = Carray):

    #global Carray

    #N1 = N+2
    m = 2.0/L

    YP = np.zeros(N+N)

    rp = np.zeros(N1)

    KF = np.concatenate([XX[:N], [0.0, 0.0]])
    rF = np.concatenate([XX[N:], [0.0, 0.0]])

    #print(KF, len(KF))
    #print(rF, len(rF))

    ca = np.zeros(N1)
    global y

    s = -0.25 * L * L / (EPS * EPS)
    rp = s * rF

    EF = cheb_Int(rp, N1)[0]
    FF = cheb_Int(EF, N1)[0]
    #print(EF)
    #print(FF)

    k1 = 0.0
    k2 = 0.0
    for i in range(2, N1):
        k1 = k1 + FF[i] * (-1.0)**(i*1.0)
        k2 = k2 + FF[i]

    FF[0] = -0.5 * (k1+k2-DELTA_V)
    FF[1] = 0.5 * (k1-k2+DELTA_V)

    #print(FF[0], FF[1])

    #print(cheb_Fx(FF, N1))

    EF = cheb_Diff(FF, N1)

    tt = np.zeros(8)
    for i in range(N):
        tt[0] = tt[0] + KF[i] * (-1.0)**(i)
        tt[1] = tt[1] + rF[i] * (-1.0)**(i)
        tt[2] = tt[2] + KF[i]
        tt[3] = tt[3] + rF[i]
        tt[4] = tt[4] + KF[i] * (i*1.0) * (i*1.0) * (-1.0)**(i+1.0)
        tt[5] = tt[5] + rF[i] * (i*1.0) * (i*1.0) * (-1.0)**(i+1.0)
        tt[6] = tt[6] + KF[i] * (i*1.0) * (i*1.0)
        tt[7] = tt[7] + rF[i] * (i*1.0) * (i*1.0)

    tE1 = 0.0
    tE2 = 0.0
    for i in range(N1):
        tE1 = tE1 + EF[i] * (-1.0)**(i)
        tE2 = tE2 + EF[i]

    a1 = 2.0 * (N1-1.0)*(N1-1.0) + (2.0*tE1-V0)
    a2 = -2.0 * (N1-2.0)*(N1-2.0) + (2.0*tE1-V0)

    b1 = 2.0*P-(ZA - 1.0) * np.exp(A_DA*V0 - ZA*DELTA_V+np.log(CA0))-tt[0]-tt[1]
    b2 = (tt[0]-tt[1]) * (2.0 * tE1-V0) - 2.0*tt[4] + 2.0 * tt[5] + np.exp(A_DA*V0 - ZA * DELTA_V + np.log(CA0)) * ((ZA*ZA - 1.0)*2.0*tE1+(1.0+ZA)*(1.0-A_DA)*V0)
    b3 = 2.0 - (ZA-1.0) * CA0 - tt[2]
    b4 = -tt[3]

    d = 0.5 * (b2-0.5*(a1-a2)*(b4+b3-b1)+a2*(b4-b3))/(a2-a1)

    KF[N1-2] = d+b3-0.5*(b4+b3-b1)
    KF[N1-1] = -d+0.5*(b4+b3-b1)
    rF[N1-2] = -d+b4
    rF[N1-1] = d

    K = cheb_Fx(KF, N1)
    r = cheb_Fx(rF, N1)
    F = cheb_Fx(FF, N1)
    EE = cheb_Fx(EF, N1)

    print(F, t)

    for i in range(N1):
        ca[i] = np.exp(ZA * (F[i]-DELTA_V)-A_DA*V0*(y[i]-1.0)+np.log(CA0))
        #print(ca[i])

    #Carray.append(ca)

    caF = cheb_Coef3(ca, N1)

    s1 = r*EE
    s2 = (K + (ZA*ZA-1.0)*ca)*EE

    #print(s1)
    #print(s2)

    EE1 = m*EE
    EEF = EF
    s1F = cheb_Coef3(s1, N1)
    s2F = cheb_Coef3(s2, N1)

    DKF = cheb_Diff(KF, N1)
    DrF = cheb_Diff(rF, N1)
    dK = cheb_Fx(DKF, N1)
    dr = cheb_Fx(DrF, N1)

    c1 = s1F + DKF + V0/m*(KF + (A_DA-1.0) * caF)
    c2 = s2F + DrF + V0/m*(rF - ZA*(A_DA-1.0)*caF)

    Dc1 = cheb_Diff(c1, N1)
    Dc2 = cheb_Diff(c2, N1)
    dKY = m*dK
    drY = m*dr

    cur = r[N-1]*EE1[N-1] + dKY[N-1]+V0*( K[N-1]+(A_DA-1.0)*ca[N-1] )

    YP[:N] = m*m*Dc1[:N]
    YP[N:N+N] = m*m*Dc2[:N]

    #A = np.matrix([res_n1[N-2:N], res_n2[N-2:N]])
    #B = [-np.sum(res_n1[:N-2] * Cxd2y[:N-2]), -np.sum(res_n2[:N-2] * Cxd2y[:N-2] )]
    #S = np.linalg.solve(A, B)
    #Cxd2y[N-2] = S[0]
    #Cxd2y[N-1] = S[1]
    return YP

if __name__ == '__main__':
    print("Start preparing...")

    z_grid = Grid(N)

    #Cx = np.zeros(N)
    #Cx = Cx + EPS
    #Cx[mth.ceil(N/2)] = C0
    #Cx[mth.ceil(N/2)-1] = C0
    #Carray = []
    #Carray.append(Cx)

    #C = cheb_Coef3(Cx, N)

    print('cheb to phys:', cheb_to_phys)
    print('phys to cheb:', phys_to_cheb)


    print("Adams...")
    #x_grid, CoefsArray = adams_bashforth(F, C, Steps, delta_x)
    t_grid = np.linspace(0.0, delta_t*Steps, Steps)

    #CoefsArray = odeint(F, C, x_grid)

    num_sol = solve_ivp(FCN, [0.0, delta_t*Steps], yF, method='LSODA', dense_output=True)
    #print(num_sol.sol(x_grid).T)
    CoefsArray = num_sol.sol(t_grid).T



    #print(solution)
    '''Karray = []
    Rarray = []
    CoefsArray=[yF]
    for i in range(Steps):
        solution = solve_ivp(FCN, [i*delta_t, (i+1)*delta_t], yF, method='LSODA')
        #print(solution.y)
        CoefsArray.append(solution)
        Karray.append(cheb_Fx(np.concatenate([CoefsArray[i][:N], [0.0, 0.0]]), N1))
        Rarray.append(cheb_Fx(np.concatenate([CoefsArray[i][N:N+N], [0.0, 0.0]]), N1))
        print(Karray)
        print(Rarray)
    print("Adams finished")'''

    #print('cheb to phys:', cheb_to_phys)
    #print('phys to cheb:', phys_to_cheb)


    Karray = []
    Rarray = []
    for i in range(len(CoefsArray)):
        Karray.append(cheb_Fx(np.concatenate([CoefsArray[i][:N], [0.0, 0.0]]), N1))
        Rarray.append(cheb_Fx(np.concatenate([CoefsArray[i][N:N+N], [0.0, 0.0]]), N1))



    Karray = np.matrix(Karray)
    Rarray = np.matrix(Rarray)
    Carray = np.matrix(Carray)


    '''fig, axs = plt.subplots(3, 1)

    axs[0] = plt.imshow(Carray.T)
    axs[1] = plt.imshow(Karray.T)
    axs[2] = plt.imshow(Rarray.T)

    fig.show()
    input()'''


    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Heatmap(
                    z=Karray.T,
                    x=t_grid,
                    y=y,
                    hoverongaps = False),
        row=1, col=1)

    fig.add_trace(go.Heatmap(
                    z=Rarray.T,
                    x=t_grid,
                    y=y,
                    hoverongaps = False),
        row=2, col=1)

    fig.add_trace(go.Heatmap(
                    z=Carray.T,
                    x=t_grid,
                    y=y,
                    hoverongaps = False),
        row=3, col=1)

    temp_name = 'Task1_plot.html'
    plot(fig, filename = temp_name, auto_open=False,
        image_width=1200,image_height=1000)
    plot(fig)

