import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
from IPython.display import Markdown as md

tf = pd.read_csv('CL_6.csv')
x = pd.DataFrame({'ell': tf['ell'], 'tau': tf['tau']})
ee = pd.DataFrame(tf['EE'])
#x = x.reshape(-1,1)
print(x)

model = PySRRegressor(
    model_selection="best",
    niterations=1000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    unary_operators=["log","exp"],
    #constraints={'^': (5, 2)},
    #complexity_of_operators={"^":1},
    #nested_constraints={"^": {"^": 2}},
    maxsize=25,
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