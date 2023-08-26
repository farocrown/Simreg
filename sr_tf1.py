# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pysr import PySRRegressor

# In [this article](https://arxiv.org/abs/2211.06393) J. Bayron Orjuela-Quintana et al. have found an explicit expression for the transfer function T(k):
# 
# $$ T(k;\omega_b,\omega_m) = [1+59.0998\ x^{1.49177}+4658.01\ x^{4.02755}+3170.79\ x^{6.06}+150.089\ x^{7.28478}]^{-\frac{1}{4}} $$
# 
# where
# $$ x=\frac{k\ Mpc}{\omega_m-\omega_b} $$
# 
# and $\omega_i=\Omega_i h^2$, where $h$ is the reduced Hubble constant and $\Omega_i$ are the density parameters where $X = b,c,m,r,\nu,\gamma$ denotes baryons, CDM, pressure-less matter, radiation, neutrinos, photons, respectively.
# 
# First of all, let's import T(k) from CLASS:

# %%
df_cl = pd.read_csv('TF_class.csv')
df_cus = pd.read_csv('TF_custom.csv')

# %% [markdown]
# Now, let's generate T(k)

# %%
def T(x):
    a = (1 + 59.0998 * x**1.49177 + 4658.01 * x**4.02755 + 3170.79 * x**6.06 + 150.089 * x**7.28478)**(-1/4)
    return a

# %%
h = 0.6781
k = np.logspace(np.log10(3 * 10 ** -5), np.log10(2),num=114,endpoint=True)
omega_b = 0.0223828 #omega baryon
omega_m = 0.1201075 #omega pressure-less matter
x = k/(omega_m-omega_b)
T_ = T(x)

# %%
plt.plot(df_cus['k (1/Mpc)'],df_cus['T(k)'], label='computed by CLASS', lw=2)
plt.plot(df_cus['k (1/Mpc)'],T(df_cus['k (1/Mpc)']/(omega_m-omega_b)), label='analityc formula (from GA)', linestyle="--", color='r')
plt.xlabel(r'$k\ [\frac{1}{Mpc}]$')
plt.ylabel(r'$T(k)$')
plt.title(r"Matter transfer function $T(k)$")
plt.loglog()
plt.grid(True)
plt.legend()
plt.show()
# %% [markdown]
# I don't understand why I obtain a good accordance while I'm neglecting that the numerator of x is adimensional ($k\ Mpc$): in fact I used in the calculation only $k$, wich is dimensional ($[k]=\frac{1}{Mpc}$).
# 
# However, let's try to fit the analytic expression using PySR:

# %%
x = x.reshape(-1,1)
cluster = ["slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc"]

# %%
model = PySRRegressor(
    model_selection="best",
    niterations=1000,  # < Increase me for better results
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
model.fit(x,T_)
print(model)

# %%
print(f"The model has this equation as output: ${model.latex()}$")

# %%
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,12))
pred = model.predict(x)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.plot(k,pred, label='analityc formula (from PySR)', lw=2)
ax1.plot(k,T_, label='analityc formula (from GA)', linestyle="--", color='r')
ax1.set_xlabel(r'$k \,\,\,\, [1/\mathrm{Mpc}]$')
ax1.set_ylabel(r'$T(k)$')
ax1.set_title('Transfer function T(k)')
ax1.legend()

ax2.set_xlabel(r'$k \,\,\,\, [1/\mathrm{Mpc}]$')
ax2.set_ylabel(r'1-GA/PySR')
ax2.plot(k,1-T_/pred)
ax2.set_title('Relative difference between the formula and data')
plt.show()


# %% [markdown]
# It seems that the equation is too difficult to find it. Let's try for steps, with equations little by little more complicated:
# n = k/(omega_m-omega_b)
# y = T(n)
# model.fit(x,y)
# md(f"The model has this equation as output: ${model.latex()}$")
# plt.plot(k,1-y/model.predict(x))
# plt.semilogy(k,model.predict(x))
# plt.semilogy(k,y)

# %% [markdown]
# # A new hope

# %% [markdown]
# Now we are generating data again with the analytic formula given by the paper, but now we are trying to replicate the choice of $k$ that they made.
# 
# (cit.) We consider that $\omega_b \in [0.0214,0.0234]$, and $\omega_m \in [0.13,0.15]$, and to see the dependence of the transfer function on these parameters, we make a grid of 4×4 pairs of $\{\omega_b, \omega_m\}$. For each considered cosmology (16 in total), we retrieve 114 points $\{k, T\}$. 

# %%
obar = np.linspace(0.0214, 0.0234, num=4, endpoint=True) #omega_b
omar = np.linspace(0.13, 0.15, num=4, endpoint=True) #omega_m

# %%
grid = [] #the pairs are (omega_b,omega_m)

for a in obar:
    line = []
    for b in omar:
        pair = (round(a, 4), round(b, 4))
        line.append(pair)
    grid.append(line)

# %% [markdown]
# Now, we calculate $x$:

# %%
k_ar = np.logspace(np.log10(3 * 10 ** -5), np.log10(2),num=114,endpoint=True)
x_1 = []
k_1 = []

for line in grid:
    for couple in line:
        a,b = couple
        x_1.append(k_ar/(b-a))
        k_1.append(k_ar)

x_1 = np.concatenate(x_1)
k_1 = np.concatenate(k_1)

# %% [markdown]
# Therefore, our preliminary dataset is composed by 1824 points. Now let's compute T(x) with the formula:

# %%
T_1 = T(x_1)

# %%
x_1 = x_1.reshape(-1,1)

# %%
model.fit(x_1,T_1)
print(model)

# %%
print(f"The model has this equation as output: ${model.latex()}$")

# %%
plt.scatter(x_1,T_1, s=1, label='From GA')
plt.scatter(x_1, model.predict(x_1), s=1, label='From PySR')
plt.loglog()
plt.legend()
plt.show()

# %%
plt.scatter(k_1,1-T_1/model.predict(x_1),s=1)
plt.show()

# %%



