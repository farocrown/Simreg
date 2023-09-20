import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor

tf = pd.read_csv('CL_20_50.csv')
x = pd.DataFrame({'ell': tf['ell'], 'tau': tf['tau']})
a = []
for i in range (20):
    a.extend(tf['EE'][0:int(len(tf)/20)])
ee = pd.DataFrame({'EE': tf['EE']/a})
#x = x.reshape(-1,1)

model = PySRRegressor(
    model_selection="best",
    niterations=10000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    unary_operators=["log","exp"],
    #constraints={'^': (2, 1)},
    #nested_constraints={"^": {"^": 2}},
    #batching=True,
    maxsize=30,
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    turbo=True,  
    #cluster_manager=cluster[1],
    #multithreading=True,
)

model.fit(x,ee)

plt.scatter(x['ell'],ee, s=1, label='CLASS data')
plt.scatter(x['ell'], model.predict(x), s=1, label='From PySR')
plt.loglog()
plt.legend()
plt.show()