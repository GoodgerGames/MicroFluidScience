from cheb import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

N = 512
Nu = 1e-4

a, b = -1.0, 1.0

def f(x):
    #return 1.0-np.exp( -x/Nu ) + np.exp( (x-1.0)/Nu )
    return x**3

def f1(x):
    return x**2

def f2(x):
    return x

x = GridAB(N, a, b)
#y = f(x)


def f(y):
    return np.sin(y)

Q = 1.0


u_coefs = np.zeros(N)
u_coefs[0] = Q/2.0

divu_coefs = np.zeros(N)
divu_coefs[0] = 1.0/ u_coefs[0]

def F(x, Ck):
    Cxd2y = np.zeros(N)
    Cxd2y = cheb_DiffP(Ck, N, 2)
    Cxd2y = cheb_Mul(Cxd2y, divu_coefs, N)
    #Cxd2y = cheb_Fx(Cxd2y, N)
    #Cxd2y = Cxd2y / cheb_Fx(u_coefs, N)

    #Cxd2y = cheb_Coef3(Cxd2y, N)
    #A = np.matrix([res_n1[N-2:N], res_n2[N-2:N]])
    #B = [-np.sum(res_n1[:N-2] * C_new[:N-2]), -np.sum(res_n2[:N-2] * C_new[:N-2] )]
    return Cxd2y


f_coefs = cheb_Coef2(f, N)
intf_coefs = cheb_Int(f_coefs, N)[0]

df_coefs = cheb_DiffP(f_coefs, N, 2)

df_values = cheb_Fx(df_coefs, N)

intf_values = cheb_Fx(intf_coefs, N)
f_values = f(x)

F_coefs = F(0, f_coefs)

#F_values = cheb_Fx(F_coefs, N)


f1_coefs = cheb_Coef2(f1, N, a, b)
f2_coefs = cheb_Coef2(f2, N, a, b)

#f_coefs = cheb_Mul(f1_coefs, f2_coefs, N)
yc = cheb_Fx(f_coefs, N)

df_coefs = cheb_DiffP(f_coefs, N, 2)
ydfc = cheb_Fx(df_coefs, N)


fig = make_subplots(rows=3, cols=1,
                    specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],)

fig.add_trace(go.Line(x=x, y=f_values, mode='lines'),
    row=1, col=1)

#fig.add_trace(go.Line(x=x, y=yc, mode='lines+markers'),
#    row=1, col=1)

fig.add_trace(go.Line(x=x, y=intf_values, mode='lines'),
    row=2, col=1)

#fig.add_trace(go.Line(x=x, y=(y-yc), mode='lines+markers'),
#    row=3, col=1)


fig.update_yaxes(title_text="Функция", row=1, col=1)
fig.update_yaxes(title_text="Производная", row=2, col=1)
fig.update_yaxes(title_text="Невязка", row=3, col=1)

print('Невязка: ', Discrepancy(f, f_coefs, N, a, b))

temp_name = 'Temp_plot.html'
plot(fig, filename = temp_name, auto_open=False,
     image_width=1200,image_height=1000)
plot(fig)
