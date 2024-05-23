from cheb import *
from adams import adams_bashforth, adams_bashforth_multiproccess, adams_bashforth_moulton

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from os import environ

from scipy.integrate import solve_ivp
from scipy.integrate import odeint

N1 = 128

N = N1-2

EPS = 1e-2

L = 1.0

delta_t = 1e-5

Steps = 10

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

ca = np.zeros(N1)
ca[:] = CA0

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


def FCN(t, XX, YP = [], ca = ca):

    #global Carray

    #print(XX)
    global y
    #N1 = N+2
    m = 2.0/L

    YP = np.zeros(N+N)

    rp = np.zeros(N1)

    KF = np.concatenate([XX[:N], [0.0, 0.0]])
    rF = np.concatenate([XX[N:], [0.0, 0.0]])



    #print('KF', KF, len(KF))
    #print('rF', rF, len(rF))

    ca = np.zeros(N1)


    s = -0.25 * L * L / (EPS * EPS)
    rp = s * rF

    EF = cheb_Int(rp, N1)[0]
    FF = cheb_Int(EF, N1)[0]
    #print('EF', EF, len(EF))
    #print('FF', FF, len(FF))

    k1 = 0.0
    k2 = 0.0
    for i in range(2, N1):
        k1 = k1 + FF[i] * (-1.0)**(i*1.0)
        k2 = k2 + FF[i]

    #print(k1, k2)

    FF[0] = -0.5 * (k1+k2-DELTA_V)
    FF[1] = 0.5 * (k1-k2+DELTA_V)

    #print(FF[0], FF[1])

    #print(cheb_Fx(FF, N1))

    EF = cheb_Diff(FF, N1)

    #print(EF)

    t1, t2, t3, t4, t5, t6, t7, t8 = .0, .0, .0, .0, .0, .0, .0, .0
    for i in range(N):
        t1 += KF[i] * (-1.0)**(i)
        t2 += rF[i] * (-1.0)**(i)
        t3 += KF[i]
        t4 += rF[i]
        t5 += KF[i] * (i) * (i) * (-1.0)**(i+1)
        t6 += rF[i] * (i) * (i) * (-1.0)**(i+1)
        t7 += KF[i] * (i) * (i)
        t8 += rF[i] * (i) * (i)

    tE1 = 0.0
    tE2 = 0.0

    for i in range(N1):
        tE1 += EF[i] * (-1.0)**(i)
        tE2 += EF[i]

    #print(tE1, tE2)

    a1 = 2.0 * (N1-1.0)*(N1-1.0) + 2.0*tE1
    a2 = -2.0 * (N1-2.0)*(N1-2.0) - 2.0*tE1

    b1 = 2.0*P-(ZA - 1.0) * CA0 * np.exp(- ZA*DELTA_V)-t1-t2
    b2 = (t1-t2) * 2.0 * tE1 - 2*t5 + 2*t6 + np.exp(- ZA * DELTA_V) * CA0 * (ZA*ZA - 1.0)*2.0*tE1
    b3 = 2.0 - (ZA-1.0) * CA0 - t3
    b4 = -t4

    #print(a1, a2)

    #print(b1, b2, b3, b4)

    #print()

    d = 0.5 * (b2-0.5*(a1-a2)*(b4+b3-b1)+a2*(b4-b3))/(a2-a1)

    #print("d", d)

    '''KFsumm1 = sum( KF[i]*(-1)**i for i in range(N) )
    rFsumm1 = sum( rF[i]*(-1)**i for i in range(N) )

    KFsumm2 = sum( KF[i] for i in range(N) )
    rFsumm2 = sum( rF[i] for i in range(N) )

    dKFsumm1 = sum( KF[i]*(i)*(i)*(-1)**(i+1) for i in range(N) )
    drFsumm1 = sum( rF[i]*(i)*(i)*(-1)**(i+1) for i in range(N) )

    dKFsumm2 = sum( KF[i]*(i)*(i) for i in range(N) )
    drFsumm2 = sum( rF[i]*(i)*(i) for i in range(N) )

    Fsumm1 = m*sum(EF[i]*(-1)**i for i in range(N1))
    Fsumm2 = m*sum(EF[i] for i in range(N1))



    A = [
        [(-1)**(N1-1), (-1)**(N1-2), (-1)**(N1-1), (-1)**(N1-2)],
        [1.0, 1.0, 1.0, 1.0],
        [ m*(N1-1)*(N1-1)*(-1)**(N1) - (-1)**(N1-1)*Fsumm1, m*(N1-2)*(N1-2)*(-1)**(N1-1) - (-1)**(N1-2)*Fsumm1, -m*(N1-1)*(N1-1)*(-1)**(N1) + (-1)**(N1-1)*Fsumm1 , -m*(N1-2)*(N1-2)*(-1)**(N1-1) + (-1)**(N1-2)*Fsumm1 ],
        [ m*(N1-1)*(N1-1) - Fsumm2, m*(N1-2)*(N1-2) - Fsumm2, -m*(N1-1)*(N1-1) + Fsumm2 , -m*(N1-2)*(N1-2) + Fsumm2 ],
         ]

    B = [ 2.0*P - KFsumm1 - rFsumm1, 2.0*P - KFsumm2 - rFsumm2, -m*dKFsumm1 + m*drFsumm1 + KFsumm1*Fsumm1 - rFsumm1*Fsumm1, -m*dKFsumm2 + m*drFsumm2 + KFsumm2*Fsumm2 - rFsumm2*Fsumm2 ]

    res1 = np.linalg.solve(A, B)
    KF[N1-2] = res1[1]
    KF[N1-1] = res1[0]
    rF[N1-2] = res1[3]
    rF[N1-1] = res1[2]'''

    KF[N1-2] = d+b3-0.5*(b4+b3-b1)
    KF[N1-1] = -d+0.5*(b4+b3-b1)
    rF[N1-2] = -d+b4
    rF[N1-1] = d


    K = cheb_Fx(KF, N1)
    r = cheb_Fx(rF, N1)
    F = cheb_Fx(FF, N1)
    EE = cheb_Fx(EF, N1)




    #print("F", F, t)
    #print('K', K, len(KF))
    #print('r', r, len(rF))



    for i in range(N1):
        ca[i] = CA0 * np.exp(ZA * (F[i]-DELTA_V))
        #print(i, (F[i]-DELTA_V), ZA * (F[i]-DELTA_V)-A_DA*V0*(y[i]-1.0)+np.log(CA0), ca[i])

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

    c1 = s1F + DKF
    c2 = s2F + DrF

    Dc1 = cheb_Diff(c1, N1)
    Dc2 = cheb_Diff(c2, N1)
    dKY = m*dK
    drY = m*dr

    cur = r[N-1]*EE1[N-1] + dKY[N-1]

    YP[:N] = m*m*Dc1[:N]
    YP[N:N+N] = m*m*Dc2[:N]

    '''fig, axs = plt.subplots(1, 5)

    axs[0].plot(cheb_Fx(rF, N1), y, label="r")
    axs[1].plot(F, y, label="F")
    axs[1].plot(EE, y, label="E")
    axs[3].plot(r, y, label="r after")
    axs[4].plot(K, y, label="K after")
    axs[0].set_title("t = " + str(t))
    axs[2].plot(ca, y, label="ca")


    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    axs[4].legend()

    plt.show()
    plt.close()'''

    print(t)

    return YP

if __name__ == '__main__':


    N_THREADS = '16'
    environ['OMP_NUM_THREADS'] = N_THREADS
    environ['OPENBLAS_NUM_THREADS'] = N_THREADS
    environ['MKL_NUM_THREADS'] = N_THREADS
    environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
    environ['NUMEXPR_NUM_THREADS'] = N_THREADS

    print("Start preparing...")

    z_grid = Grid(N)

    fig = make_subplots(rows=5, cols=1)

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
    t_grid = np.linspace(0.0, delta_t*Steps, Steps)

    #methods: 'LSODA', 'RK45', 'BDF'
    num_sol = solve_ivp(
        FCN,
        [0.0, delta_t*Steps],
        yF,
        method='BDF',
        dense_output=True
        )
    CoefsArray = num_sol.sol(t_grid).T

    F_values = FCN(0.0, yF)[0]
    rho_values = cheb_Fx(np.concatenate([yF[N:], [0.0, 0.0]]), N1 )
    #print(F_values)

    #print(solution)
    '''Karray = []
    Rarray = []
    Carray = []
    CoefsArray=[yF]
    for i in range(Steps):
        print("STEP ", i, ": t = ", i*delta_t)
        solution = solve_ivp(FCN, [i*delta_t, (i+1)*delta_t], yF, method='BDF')
        CoefsArray.append(solution.y.T[-1])

        Karray.append(cheb_Fx(np.concatenate((CoefsArray[i][:N], [0.0, 0.0])), N1))
        Rarray.append(cheb_Fx(np.concatenate((CoefsArray[i][N:N+N], [0.0, 0.0])), N1))

        yF = FCN(i*delta_t, yF)
        Carray.append(ca)
        fig1, axs = plt.subplots(1, 5)

        axs[0].plot(Karray[i], y, label="K")
        axs[1].plot(Rarray[i], y, label="R")
        axs[2].plot(Carray[i], y, label="Ca")

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        plt.show()'''






    #print('cheb to phys:', cheb_to_phys)
    #print('phys to cheb:', phys_to_cheb)
    #print(yF)
    #CoefsArray = adams_bashforth(FCN, yF, Steps, delta_t)[1]

    print("Adams finished")

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


    fig = make_subplots(rows=5, cols=1)

    fig.add_trace(go.Heatmap(
                    z=Karray.T,
                    x=t_grid,
                    y=y,
                    hoverongaps = False,
                    coloraxis = "coloraxis",
                    ),
        row=1, col=1)

    fig.add_trace(go.Heatmap(
                    z=Rarray.T,
                    x=t_grid,
                    y=y,
                    hoverongaps = False,
                    coloraxis = "coloraxis",
                    ),
        row=2, col=1)

    fig.add_trace(go.Heatmap(
                    z=Carray.T,
                    x=t_grid,
                    y=y,
                    hoverongaps = False,
                    coloraxis = "coloraxis",
                    ),
        row=3, col=1)

    #fig.add_trace(go.Line(x=y, y=F_values, mode='lines'),
    #    row=4, col=1)

    #fig.add_trace(go.Line(x=y, y=rho_values, mode='lines'),
    #    row=5, col=1)

    fig.update_yaxes(title_text="K", row=1, col=1)
    fig.update_yaxes(title_text="rho", row=2, col=1)
    fig.update_yaxes(title_text="Ca", row=3, col=1)
    fig.update_yaxes(title_text="F", row=4, col=1)
    fig.update_yaxes(title_text="rho0", row=5, col=1)

    fig.update_layout(coloraxis = {'colorscale':'viridis'})

    temp_name = 'Task1_plot.html'
    plot(fig, filename = temp_name, auto_open=False,
        image_width=1200,image_height=1000)
    plot(fig)

