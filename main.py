from cheb import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

N = 100

def f(x):
    return np.sin(5.0*x)

def df(x):
    return 5.0 * np.cos(5.0*x)

z = []

for i in range(N):
    z.append(mth.cos( (N-i-1)/(N-1) * mth.pi ))
z = np.array(z)

y = f(z)
ydf = df(z)

f_coefs = cheb_Coef(z, f, N)
yc = []
for i in range(N):
    yc.append(cheb_Fx(f_coefs, z[i], N))

df_coefs = cheb_Diff(f_coefs, N)
ydfc = []
for i in range(N):
    ydfc.append(cheb_Fx(df_coefs, z[i], N))


fig = make_subplots(rows=2, cols=1,
                    specs=[[{"type": "xy"}], [{"type": "xy"}]],)

fig.add_trace(go.Line(x=z, y=y, mode='lines'),
    row=1, col=1)

fig.add_trace(go.Line(x=z, y=yc, mode='lines+markers'),
    row=1, col=1)

fig.add_trace(go.Line(x=z, y=ydf, mode='lines'),
    row=2, col=1)

fig.add_trace(go.Line(x=z, y=ydfc, mode='lines+markers'),
    row=2, col=1)


fig.update_yaxes(title_text="y", row=1, col=1)
fig.update_yaxes(title_text="y", range=[-6, 6], row=2, col=1)


temp_name = 'Temp_plot.html'
plot(fig, filename = temp_name, auto_open=False,
     image_width=1200,image_height=1000)
plot(fig)
