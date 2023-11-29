import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor

#for f(l) you will need Ratio(l).csv
#for g(t) you will need Ratio(t)-l6.csv

tf = pd.read_csv('FINALE-ELL-NODIV-E.csv')
x = pd.DataFrame({'ell': tf['ell']})
ee = pd.DataFrame(tf['EE/EE_ref'])
#x = x.reshape(-1,1)

#for f(l) you will need log and exp
#for g(t) only binary operators

model = PySRRegressor(
    model_selection="best",
    niterations=5000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    unary_operators=["exp"], #"log",'sinh','cosh','tanh','sin','cos','tan'],
    constraints={'^': (2, 2)},
    nested_constraints={"^": {"^": 2}},
    maxsize=20,
    loss="loss(prediction, target) = ((prediction - target)^2) / (target)^2",
    # ^ Custom loss function (julia syntax)
    turbo=True,  
    #cluster_manager=cluster[1],
    #multithreading=True,
)

model.fit(x,ee)

plt.scatter(x,ee, s=1, label='CLASS data')
plt.scatter(x, model.predict(x), s=1, label='From PySR')
plt.loglog()
plt.legend()
plt.show()