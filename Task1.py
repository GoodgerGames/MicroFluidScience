from cheb import *
from adams import adams_bashforth, adams_bashforth_multiproccess, adams_bashforth_moulton

import os
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

EPS = 1e-1

L = 1.0

delta_t = 1e-5

Steps = 1000
STP = 10

K0 = np.zeros(N1)
R0 = np.zeros(N1)

NU = 1e-3
P = 1.0
ZA = 2.0
A_DA = 1.0
CA0 = 1e-3
DELTA_V = 7.0
V0 = 5.0

K0[:] = 2.0 - ZA*CA0

z = Grid(N1)

y = 0.5 * L * (z+1.0)

yF = np.zeros(N*3)
yp = np.zeros(N*3)

ca = np.zeros(N1)
ca[:] = CA0

Rcoefs = cheb_Coef3(R0, N1)
Kcoefs = cheb_Coef3(K0, N1)
yF[:N] = Kcoefs[:N]
yF[N:2*N] = Rcoefs[:N]
yF[2*N:3*N]=cheb_Coef3(ca, N1)[:N]
neqn = 2*N
cur = 0.0

Carray = []

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def FCN(t, XX, YP = [], ca = ca):
    print(t)

    global y
    m = 2.0/L

    YP = np.zeros(N+N+N)

    rp = np.zeros(N1)

    KF = np.concatenate([XX[:N], [0.0, 0.0]])
    rF = np.concatenate([XX[N:2*N], [0.0, 0.0]])
    CaF = np.concatenate([XX[2*N:], [0.0, 0.0]])
    Ca = cheb_Fx(CaF, N1)

    #s = -0.25 * L * L / (EPS * EPS)
    rp = -(rF-ZA*CaF)/NU/NU

    EF = cheb_Int(rp, N1)[0]
    FF = cheb_Int(EF, N1)[0]

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

    '''t1, t2, t3, t4, t5, t6, t7, t8 = .0, .0, .0, .0, .0, .0, .0, .0
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

    a1 = 2.0 * (N1-1.0)*(N1-1.0) + (2.0*tE1-V0)
    a2 = -2.0 * (N1-2.0)*(N1-2.0) - (2.0*tE1-V0)

    #b1 = 2.0*P-(ZA - 1.0) * CA0 * np.exp(- ZA*DELTA_V)-t1-t2
    b1 = 2.0*P-t1-t2
    b2 = (t1-t2) * (2.0 * tE1-V0) - 2*t5 + 2*t6
    b3 = 2.0 - t3
    b4 = -t4

    #print(a1, a2)

    #print(b1, b2, b3, b4)

    #print()

    d = 0.5 * (b2-0.5*(a1-a2)*(b4+b3-b1)+a2*(b4-b3))/(a2-a1)'''


    KFsumm1 = sum( KF[i]*(-1)**i for i in range(N) )
    rFsumm1 = sum( rF[i]*(-1)**i for i in range(N) )

    KFsumm2 = sum( KF[i] for i in range(N) )
    rFsumm2 = sum( rF[i] for i in range(N) )

    dKFsumm1 = sum( KF[i]*(i)*(i)*(-1)**(i+1) for i in range(N) )
    drFsumm1 = sum( rF[i]*(i)*(i)*(-1)**(i+1) for i in range(N) )

    dKFsumm2 = sum( KF[i]*(i)*(i) for i in range(N) )
    drFsumm2 = sum( rF[i]*(i)*(i) for i in range(N) )

    Fsumm1 = m*sum(EF[i]*(-1)**i for i in range(N1))
    Fsumm2 = m*sum(EF[i] for i in range(N1))

    CaFsumm1 = sum( CaF[i]*(-1)**i for i in range(N) )
    CaFsumm2 = sum( CaF[i] for i in range(N) )
    dCaFsumm1 = sum( CaF[i]*(i)*(i)*(-1)**(i+1) for i in range(N) )
    dCaFsumm2 = sum( CaF[i]*(i)*(i) for i in range(N) )

    A = [
        [m*(N1-1)*(N1-1)*(-1)**(N1) - ZA*(-1)**(N1-1)*Fsumm1, m*(N1-2)*(N1-2)*(-1)**(N1-1) - ZA*(-1)**(N1-2)*Fsumm1],
        [1.0, 1.0],
    ]
    B = [-m*dCaFsumm1 + ZA*CaFsumm1*Fsumm1, CA0 - CaFsumm2]

    #A = np.matrix(A)
    res1 = np.linalg.solve(A, B)
    CaF[N1-2] = res1[1]
    CaF[N1-1] = res1[0]

    CaFsumm1 = sum(CaF[i]*(-1)**i for i in range(N1))
    CaFsumm2 = sum(CaF[i] for i in range(N1))


    A = [
        [(-1)**(N1-1), (-1)**(N1-2), (-1)**(N1-1), (-1)**(N1-2)],
        [0.0, 0.0, 1.0, 1.0],
        [ m*(N1-1)*(N1-1)*(-1)**(N1) - Fsumm1*(-1)**(N1-1), m*(N1-2)*(N1-2)*(-1)**(N1-1) - Fsumm1*(-1)**(N1-2), -m*(N1-1)*(N1-1)*(-1)**(N1) + Fsumm1*(-1)**(N1-1) , -m*(N1-2)*(N1-2)*(-1)**(N1-1) + Fsumm1*(-1)**(N1-2) ],
        [ 1.0, 1.0, 1.0, 1.0],
         ]

    B = [ 2.0*P - KFsumm1 - rFsumm1, -rFsumm2 + ZA*CaFsumm2, -m*dKFsumm1 + m*drFsumm1 + KFsumm1*Fsumm1 - rFsumm1*Fsumm1, 2.0 - KFsumm2 - rFsumm2 ]

    res1 = np.linalg.solve(A, B)
    KF[N1-2] = res1[1]
    KF[N1-1] = res1[0]
    rF[N1-2] = res1[3]
    rF[N1-1] = res1[2]

    '''KF[N1-2] = d+b3-0.5*(b4+b3-b1)
    KF[N1-1] = -d+0.5*(b4+b3-b1)
    rF[N1-2] = -d+b4
    rF[N1-1] = d'''

    K = cheb_Fx(KF, N1)
    r = cheb_Fx(rF, N1)
    F = cheb_Fx(FF, N1)
    Ca = cheb_Fx(CaF, N1)
    EE = cheb_Fx(EF, N1)

    s1 = r*EE
    s2 = K*EE
    s3 = -Ca*EE * ZA

    EE1 = m*EE
    EEF = EF
    s1F = cheb_Coef3(s1, N1)
    s2F = cheb_Coef3(s2, N1)
    s3F = cheb_Coef3(s3, N1)

    DKF = cheb_Diff(KF, N1)
    DrF = cheb_Diff(rF, N1)
    DCaF = cheb_Diff(CaF, N1)
    dK = cheb_Fx(DKF, N1)
    dr = cheb_Fx(DrF, N1)
    dCa = cheb_Fx(DCaF, N1)

    c1 = s1F + DKF + KF * V0/m
    c2 = s2F + DrF + rF * V0/m
    c3 = s3F + DCaF + CaF * A_DA*V0/m

    Dc1 = cheb_Diff(c1, N1)
    Dc2 = cheb_Diff(c2, N1)
    Dc3 = cheb_Diff(c3, N1)

    YP[:N] = m*m*Dc1[:N]
    YP[N:N+N] = m*m*Dc2[:N]
    YP[N+N:] = m*m/A_DA*Dc3[:N]

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

    fig = make_subplots(rows=3, cols=1)

    print('cheb to phys:', cheb_to_phys)
    print('phys to cheb:', phys_to_cheb)


    print("Adams...")
    t_grid = np.linspace(0.0, delta_t*Steps, Steps)

    #methods: 'LSODA', 'RK45', 'BDF'
    '''num_sol = solve_ivp(
        FCN,
        [0.0, delta_t*Steps],
        yF,
        method='BDF',
        dense_output=True
        )
    CoefsArray = num_sol.sol(t_grid).T'''

    my_path = os.path.abspath(__file__)
    Karray = []
    Rarray = []
    Carray = []
    CoefsArray=[yF]

    Karray.append(cheb_Fx(np.concatenate((CoefsArray[-1][:N], [0.0, 0.0])), N1))
    Rarray.append(cheb_Fx(np.concatenate((CoefsArray[-1][N:N+N], [0.0, 0.0])), N1))
    Carray.append(cheb_Fx(np.concatenate((CoefsArray[-1][N+N:N+N+N], [0.0, 0.0])), N1))

    #Karray.append(cheb_Fx(CoefsArray[-1][:N], N1))
    #Rarray.append(cheb_Fx(CoefsArray[-1][N:N+N], N1))
    #Carray.append(cheb_Fx(CoefsArray[-1][N+N:N+N+N], N1))

    plt.subplot(1, 2, 1)

    ca_plot = plt.plot(y, Carray[-1], label="Ca", alpha=0.8, linewidth=2.5, color='black')
    cplus_plot = plt.plot(y, (Karray[-1]+Rarray[-1])/2.0, label="C+", alpha=0.8, linewidth=2.5, color='red')
    cminus_plot = plt.plot(y, (Karray[-1]-Rarray[-1])/2.0, label="C-", alpha=0.8, linewidth=2.5, color='blue')

    plt.legend()
    plt.ylim(0.0, 3.0)
    plt.tight_layout()

    plt.subplot(1, 2, 2)

    ca_plot2 = plt.plot(y, Carray[-1], label="Ca", alpha=0.8, linewidth=2.5, color='black')
    r_plot = plt.plot(y, Rarray[-1], label="rho", alpha=0.8, linewidth=2.5, color='green')
    k_plot = plt.plot(y, Karray[-1], label="K", alpha=0.8, linewidth=2.5, color='purple')

    plt.suptitle("t = " + str(0.0*delta_t))
    plt.legend()
    plt.ylim(0.0, 3.0)
    plt.tight_layout()

    dir = 'deltaV__'+str(DELTA_V).replace('.', '_')+'__V0__'+str(V0).replace('.', '_')+'__ZA__'+str(ZA).replace('.', '_')+'__ALPHA__'+str(A_DA).replace('.', '_')
    mkdir_p(dir)
    plt.savefig(dir+'/t'+str(0).replace('.', '_')+'.png')
    plt.close()

    for i in range(Steps):
        print("STEP ", i+1, ": t = ", (i+1)*delta_t)
        #methods: 'LSODA', 'RK45', 'BDF'
        solution = solve_ivp(FCN, [i*delta_t, (i+1)*delta_t], CoefsArray[-1], method='LSODA')
        CoefsArray.append(solution.y.T[-1])

        Karray.append(cheb_Fx(np.concatenate((CoefsArray[-1][:N], [0.0, 0.0])), N1))
        Rarray.append(cheb_Fx(np.concatenate((CoefsArray[-1][N:N+N], [0.0, 0.0])), N1))
        Carray.append(cheb_Fx(np.concatenate((CoefsArray[-1][N+N:N+N+N], [0.0, 0.0])), N1))

        #Karray.append(cheb_Fx(CoefsArray[-1][:N], N1))
        #Rarray.append(cheb_Fx(CoefsArray[-1][N:N+N], N1))
        #Carray.append(cheb_Fx(CoefsArray[-1][N+N:N+N+N], N1))

        if i%STP == 0:
            plt.subplot(1, 2, 1)

            ca_plot = plt.plot(y, Carray[-1], label="Ca", alpha=0.8, linewidth=2.5, color='black')
            cplus_plot = plt.plot(y, (Karray[-1]+Rarray[-1])/2.0, label="C+", alpha=0.8, linewidth=2.5, color='red')
            cminus_plot = plt.plot(y, (Karray[-1]-Rarray[-1])/2.0, label="C-", alpha=0.8, linewidth=2.5, color='blue')
            plt.legend()
            plt.ylim(0.0, 3.0)
            plt.tight_layout()

            plt.subplot(1, 2, 2)

            ca_plot2 = plt.plot(y, Carray[-1], label="Ca", alpha=0.8, linewidth=2.5, color='black')
            r_plot = plt.plot(y, Rarray[-1], label="rho", alpha=0.8, linewidth=2.5, color='green')
            k_plot = plt.plot(y, Karray[-1], label="K", alpha=0.8, linewidth=2.5, color='purple')

            plt.suptitle("t = " + str((i+1)*delta_t))
            plt.legend()
            plt.ylim(0.0, 3.0)
            plt.tight_layout()

            plt.savefig(dir+'/t'+str(i+1).replace('.', '_')+'.png')
            plt.close()






    '''CoefsArray = adams_bashforth(FCN, yF, Steps, delta_t)[1]
    Karray = []
    Rarray = []
    for i in range(len(CoefsArray)):
        Karray.append(cheb_Fx(np.concatenate([CoefsArray[i][:N], [0.0, 0.0]]), N1))
        Rarray.append(cheb_Fx(np.concatenate([CoefsArray[i][N:N+N], [0.0, 0.0]]), N1))
        Carray.append(cheb_Fx(np.concatenate([CoefsArray[i][N+N:N+N+N], [0.0, 0.0]]), N1))'''



    print("Adams finished")


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
                    #coloraxis = "coloraxis",
                    ),
        row=1, col=1)

    fig.add_trace(go.Heatmap(
                    z=Rarray.T,
                    x=t_grid,
                    y=y,
                    hoverongaps = False,
                    #coloraxis = "coloraxis",
                    ),
        row=2, col=1)

    fig.add_trace(go.Heatmap(
                    z=Carray.T,
                    x=t_grid,
                    y=y,
                    hoverongaps = False,
                    #coloraxis = "coloraxis",
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

