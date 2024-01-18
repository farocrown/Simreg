import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
import os

script_folder = os.path.dirname(os.path.abspath(__file__)) #Script directory path
data_folder = os.path.join(script_folder, "..", "Data") #Data directory path
file_path = os.path.join(data_folder, "TF_64.csv") #File path

tf = pd.read_csv(file_path)
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



