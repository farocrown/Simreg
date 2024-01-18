import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
from IPython.display import Markdown as md

tf = pd.read_csv('TF_64.csv')
z = pd.DataFrame(tf['k (h/Mpc)']/tf['omega_m'])
x = pd.DataFrame(tf['x'])
y = pd.DataFrame({'k': tf['k (h/Mpc)'], 'omega_b': tf['omega_b'],'omega_m': tf['omega_m'],'omega_nu': tf['omega_nu']})
T = pd.DataFrame(tf['T(k)'])
logT = pd.DataFrame(tf['log10(T(k))'])


model = PySRRegressor(
    model_selection="best",
    niterations=10000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    #unary_operators=['log10'],
    #constraints={'^': (5, 2)},
    #complexity_of_operators={"^":2},
    nested_constraints={"^": {"^": 2}},
    maxsize=25,
    loss="loss(prediction, target) =  ((prediction - target)^2) / (target^2)",
    # ^ Custom loss function (julia syntax)
    turbo=True,  
)

model.fit(x,T)

plt.scatter(x[0:114],T[0:114], s=1, label='CLASS data')
plt.plot(x[0:114], model.predict(x)[0:114], color='r', label='From PySR')
plt.loglog()
plt.legend()
plt.show()



