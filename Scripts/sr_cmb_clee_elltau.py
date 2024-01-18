import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
import os

script_folder = os.path.dirname(os.path.abspath(__file__)) #Script directory path
data_folder = os.path.join(script_folder, "..", "Data") #Data directory path
file_path = os.path.join(data_folder, "CL_20_2.csv") #File path

tf = pd.read_csv(file_path)
x = pd.DataFrame({'ell': tf['ell'][0:2998], 'tau': tf['tau'][0:2998]})
a = []
for i in range (20):
    a.extend(tf['EE'][0:int(len(tf)/20)])
ee = pd.DataFrame({'EE': tf['EE'][0:2998]})
#ee = pd.DataFrame({'EE': np.exp(-2*x['tau'][0])*tf['EE']/(a*np.exp(-2*x['tau']))})
#x = x.reshape(-1,1)

model = PySRRegressor(
    model_selection="best",
    niterations=10000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    unary_operators=["log","exp",'sinh','cosh','tanh','sin','cos','tan'],
    #complexity_of_operators={"log":1,"exp":1,'sinh':1,'cosh':1,'tanh':1,'sin':1,'cos':1,'tan':1},
    #constraints={'^': (2, 2)},
    #nested_constraints={"^": {"^": 2}},
    maxsize=20,
    loss="loss(prediction, target) = ((prediction - target)^2 / (target)^2)",
    # ^ Custom loss function (julia syntax)
    turbo=True,  
    #cluster_manager='slurm',
    #multithreading=True,
)

model.fit(x,ee)

plt.scatter(x['ell'],ee, s=1, label='CLASS data')
plt.scatter(x['ell'], model.predict(x), s=1, label='From PySR')
plt.loglog()
plt.legend()
plt.show()