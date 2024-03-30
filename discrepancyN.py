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

NN = [10+i*8 for i in range(200)]
DN = []

for N in NN:
    x = GridAB(N, a, b)
    y = f(x)
    f_coefs = cheb_Coef2(f, N, a, b)
    yc = cheb_Fx(f_coefs, N)

    DN.append(Discrepancy(f, f_coefs, N, a, b))
    #print(DN)

fig = px.line(x=NN, y=DN)

fig.update_yaxes(title_text="Зависимость Невязки от N")

temp_name = 'Temp_plot_Discrepancy.html'
plot(fig, filename = temp_name, auto_open=False,
     image_width=1200,image_height=1000)
plot(fig)





