from cheb import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

N = 216
Nu = 1e-2

a, b = 0.0, 1.0

def f(x):
    return 1.0-np.exp( -x/Nu ) + np.exp( (x-1.0)/Nu )
    #return x*x

x = GridAB(N, a, b)
print(x)
y = f(x)

f_coefs = cheb_Coef2(f, N, a, b)
yc = cheb_Fx(f_coefs, N)

df_coefs = cheb_Diff(f_coefs, N)
ydfc = cheb_Fx(df_coefs, N)


fig = make_subplots(rows=2, cols=1,
                    specs=[[{"type": "xy"}], [{"type": "xy"}]],)

fig.add_trace(go.Line(x=x, y=y, mode='lines'),
    row=1, col=1)

fig.add_trace(go.Line(x=x, y=yc, mode='lines+markers'),
    row=1, col=1)

fig.add_trace(go.Line(x=x, y=ydfc, mode='lines+markers'),
    row=2, col=1)


fig.update_yaxes(title_text="y", row=1, col=1)
fig.update_yaxes(title_text="y", row=2, col=1)


temp_name = 'Temp_plot.html'
plot(fig, filename = temp_name, auto_open=False,
     image_width=1200,image_height=1000)
plot(fig)
