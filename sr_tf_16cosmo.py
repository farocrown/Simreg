import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
from IPython.display import Markdown as md

tf = pd.read_csv('TF_16.csv')
z = pd.DataFrame(tf['k (h/Mpc)']/tf['omega_m'])
x = pd.DataFrame(tf['x'])
y = pd.DataFrame({'k': tf['k (h/Mpc)'], 'omega_b': tf['omega_b'],'omega_m': tf['omega_m']})
T = pd.DataFrame(tf['T(k)'])
logT = pd.DataFrame(tf['log10(T(k))'])
#x = x.reshape(-1,1)

model = PySRRegressor(
    model_selection="best",
    niterations=100,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    unary_operators=['log10'],
    constraints={'^': (5, 2)},
    #complexity_of_operators={"^":1},
    #nested_constraints={"^": {"^": 2}},
    maxsize=20,
    loss="loss(prediction, target) = ((prediction - target)^2)", # / (target^2)",
    # ^ Custom loss function (julia syntax)
    turbo=True,  
    #cluster_manager=cluster[1],
    #multithreading=True,
)

#model.fit(x,T)
model.fit(x,logT)

#plt.scatter(x,T, s=1, label='CLASS data')
#plt.scatter(x, model.predict(x), s=1, label='From PySR')
plt.scatter(x,T, s=1, label='CLASS data')
plt.scatter(x, model.predict(x), s=1, label='From PySR')
plt.loglog()
plt.legend()
plt.show()



