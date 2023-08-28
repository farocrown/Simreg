# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor
from IPython.display import Markdown as md

# %% [markdown]
# This file pretends to fit data from CLASS to obtain an analytic formula for T(k). Now let's import T(k):

# %%
tf = pd.read_csv('TF_16cosmo.csv')
k = tf['k (h/Mpc)'].values
x = tf['x'].values
T = tf['T(k)'].values

# %%
x = x.reshape(-1,1)

# %% [markdown]
# Let's define a Regressor:

# %%
model = PySRRegressor(
    model_selection="best",
    niterations=10000,  # < Increase me for better results
    binary_operators=["+","*","-","/","^"],
    constraints={'^': (2, 1)},
    nested_constraints={"^": {"^": 2}},
    maxsize=20,
    loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    turbo=True,  
    #cluster_manager=cluster[1],
    #multithreading=True,
)

# %%
model.fit(x,T)

# %%
print(f"The model has this equation as output: ${model.latex()}$")

# %%
plt.scatter(x,T, s=1, label='CLASS data')
plt.scatter(x, model.predict(x), s=1, label='From PySR')
plt.loglog()
plt.legend()
plt.show()

# %%



