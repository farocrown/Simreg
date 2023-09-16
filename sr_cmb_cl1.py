import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor

#for f(l) you will need Ratio(l).csv
#for g(t) you will need Ratio(t)-l6.csv

tf = pd.read_csv('Ratio(t)-l6.csv')
x = pd.DataFrame({'tau': tf['tau']})
ee = pd.DataFrame(tf['EE/EE_ref'])
#x = x.reshape(-1,1)

#for f(l) you will need log and exp
#for g(t) only binary operators

model = PySRRegressor(
    model_selection="best",
    niterations=1000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    #unary_operators=["log","exp"],
    constraints={'^': (2, 1)},
    nested_constraints={"^": {"^": 2}},
    maxsize=25,
    loss="loss(prediction, target) = (prediction - target)^2",
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