# Cosmological application of symbolic regression algorithms 
This repository contains the code that I implemented for my bachelor degree. My project aims to get simple and closed analityc formulae for the Matter Transfer Function, that we will call $T(k)$ and the power spectrum of CMB polarization anisotropies, namely $C_{\ell}^{EE}$.

### Datasets 
In the 'Notebook' folder, you can find two files that generate the datasets:
1. **tf_CLASS_Data.ipynb** is a jupyter notebook to compute $T(k)$ as a function of the scale $k$, and the reduced density parameters of baryons, matter, and massive neutrinos ($\omega_i$). It will create four datasets:
   - *TF_16.csv* and *TF_16_test.csv*, where the reduced density parameters of baryons and cold dark matter are varying,
   - *TF_64.csv* and *TF_16_test.csv*, where also the mass of one massive neutrino varies.
2. **tf_CLASS_Data.ipynb** is a jupyter notebook to compute $C_{\ell}^{EE}$ as a function of the multipole $\ell$, and the optical depth at reionization $\tau_{reio}$. It will create four datasets:
   - *TF_16.csv* and *TF_16_test.csv*, where the reduced density parameters of baryons and cold dark matter are varying,
    





