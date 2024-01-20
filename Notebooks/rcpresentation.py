import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 7)  # Larghezza x Altezza 
plt.rcParams['lines.linewidth'] = 1.6  # Spessore delle linee predefinito
#plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.titlesize'] = 12
plt.rcParams.update({
    "font.family": "serif",            # Usa un carattere con grazie (serif)
    "font.serif": ["Times New Roman"], # Specifica il font da utilizzare
    "axes.labelsize": 25,              # Dimensione delle etichette sugli assi
    "xtick.labelsize": 25,             # Dimensione delle etichette dell'asse X
    "ytick.labelsize": 25,             # Dimensione delle etichette dell'asse Y
    'legend.fontsize': 15,
    "mathtext.fontset": "cm"
})
plt.rcParams.update({
    "scatter.marker": "o",        # Tipo di marker (cerchio)
    "lines.markersize": 8,       # Grandezza dei marker (valore in punti)
})

plt.rcParams['legend.loc'] = 'best'  # Posizione della legenda


