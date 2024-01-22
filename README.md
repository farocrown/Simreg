# Cosmological application of symbolic regression algorithms 
This repository contains the code that I implemented for my bachelor degree. My project aims to get simple and closed analityc formulae for the Matter Transfer Function, that we will call $T(k)$ and the power spectrum of CMB polarization anisotropies, namely $C_{\ell}^{EE}$.

### Datasets 
In the 'Notebooks' folder, you can find two files that generate the datasets:
1. **tf_CLASS_Data.ipynb** is a jupyter notebook to compute $T(k)$ as a function of the scale $k$ and the reduced density parameters of baryons, matter, and massive neutrinos ($\omega_i$). It will create the datasets:
   - *TF_16\*.csv*, where the reduced density parameters of baryons and cold dark matter are varying,
   - *TF_64\*.csv*, where also the mass of one massive neutrino varies.
2. **cl_CLASS_Data.ipynb** is a jupyter notebook to compute $C_{\ell}^{EE}$ as a function of the multipole $\ell$ and the optical depth at reionization $\tau_{reio}$. It will create the datasets:
   - *CL_20\*.csv*, where $\tau_{reio}$ is varying and $\ell$ spans in $[2,3000]$,
   - *CL_20_50\*.csv*, where $\tau_{reio}$ is varying and $\ell$ spans in $[2,50]$.

### Scripts
In the 'Scripts' folder, you can find several files that implement Symbolic Regression (SR) on the datasets:
1. **sr_tf\*.py** are scripts to apply SR to $T(k)$ in both cosmological scenarios (with or without massive neutrinos),
2. **sr_cmb\*.py** are scripts to apply SR to $C_{\ell}^{EE}$. In the first case I adopted a 'separation of variables' approach, in the second case I made a 'two-variable' regression.

### Data visualization
In the 'Notebooks' folder, you can find two files to visualize the results:
1. **sr_tf_graphs.ipynb**, a notebook to visualize the results of the fit and some consideration on the errors, regarding $T(k)$,
2. **sr_cmb_cl_graphs.ipynb** does the same, but for $C_{\ell}^{EE}$.
   
A copy of each graph is saved in the 'Images' folder.

### Validation tests
In the 'Notebooks' folder, you can find two file to test the algorithms outside their range of training:
1. **sr_tf_test_hypercube.ipynb** a notebook that tests the expression for $T(k)$,
2. **sr_cmb_extrapolation.ipynb** a notebook that proposes a more interesting formula for $C_{\ell}^{EE}$.

Feel free to explore, analyze, and contribute to the further development of this project. If you have any questions or suggestions, please don't hesitate to reach out!


