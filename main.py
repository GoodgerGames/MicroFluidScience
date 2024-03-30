from cheb import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

N = 512
Nu = 1e-4

a, b = 0.0, 1.0

def f(x):
    return 1.0-np.exp( -x/Nu ) + np.exp( (x-1.0)/Nu )

x = GridAB(N, a, b)
y = f(x)

f_coefs = cheb_Coef2(f, N, a, b)
yc = cheb_Fx(f_coefs, N)

df_coefs = cheb_Diff(f_coefs, N, a, b)
ydfc = cheb_Fx(df_coefs, N)


fig = make_subplots(rows=3, cols=1,
                    specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],)

fig.add_trace(go.Line(x=x, y=y, mode='lines'),
    row=1, col=1)

fig.add_trace(go.Line(x=x, y=yc, mode='lines+markers'),
    row=1, col=1)

fig.add_trace(go.Line(x=x, y=ydfc, mode='lines+markers'),
    row=2, col=1)

fig.add_trace(go.Line(x=x, y=(y-yc), mode='lines+markers'),
    row=3, col=1)


fig.update_yaxes(title_text="Функция", row=1, col=1)
fig.update_yaxes(title_text="Производная", row=2, col=1)
fig.update_yaxes(title_text="Невязка", row=3, col=1)

print('Невязка: ', Discrepancy(f, f_coefs, N, a, b))

temp_name = 'Temp_plot.html'
plot(fig, filename = temp_name, auto_open=False,
     image_width=1200,image_height=1000)
plot(fig)
