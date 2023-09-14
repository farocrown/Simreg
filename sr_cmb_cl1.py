import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
from IPython.display import Markdown as md

tf = pd.read_csv('CL_6_50_sort.csv')
x = pd.DataFrame({'ell': tf['ell'][0:6], 'tau': tf['tau'][0:6]})
ee = pd.DataFrame(tf['EE'][0:6])
#x = x.reshape(-1,1)

model = PySRRegressor(
    model_selection="best",
    niterations=1000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    #unary_operators=["log","exp","sin","cos"],
    maxsize=10,
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    #turbo=True,  
    #cluster_manager=cluster[1],
    #multithreading=True,
)

model.fit(x,ee)

plt.scatter(x['ell'],ee, s=1, label='CLASS data')
plt.scatter(x['ell'], model.predict(x), s=1, label='From PySR')
plt.loglog()
plt.legend()
plt.show()