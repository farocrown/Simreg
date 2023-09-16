import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
from IPython.display import Markdown as md

tf = pd.read_csv('TF_64.csv')
k = np.array(tf['k (h/Mpc)'])
z = np.array(tf['k (h/Mpc)']/tf['omega_m'])
x = np.array(tf['x'])
y = np.array([tf['k (h/Mpc)'],tf['omega_b'],tf['omega_m']])
T = np.array(tf['T(k)'])
x = x.reshape(-1,1)
z = z.reshape(-1,1)
k = k.reshape(-1,1)

model = PySRRegressor(
    model_selection="best",
    niterations=1000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    constraints={'^': (5, 2)},
    complexity_of_operators={"^":1},
    nested_constraints={"^": {"^": 2}},
    maxsize=25,
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    turbo=True,  
    #cluster_manager=cluster[1],
    #multithreading=True,
)
x_ax = z
model.fit(x_ax,T)

plt.scatter(x_ax,T, s=1, label='CLASS data')
plt.scatter(x_ax, model.predict(x_ax), s=1, label='From PySR')
plt.loglog()
plt.legend()
plt.show()



