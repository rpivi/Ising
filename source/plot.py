import matplotlib.pyplot as plt
import numpy as np  
from scipy.optimize import curve_fit


#plot of 2 observables vs Monte Carlo steps
def plot_two_steps(obs1, obs2, name="Observable", name1="Observable 1", name2="Observable 2", BJ=0):
    T = 1/BJ if BJ != 0 else "Unknown"
    plt.figure(figsize=(8,6))
    plt.plot(obs1, label=name1)
    plt.plot(obs2, label=name2)
    plt.grid()
    plt.title(f"{name} vs Monte Carlo time at T={T}")
    plt.xlabel("t [Monte Carlo]")
    plt.ylabel(name)
    plt.legend([f"{name1}", f"{name2}"])
    #save figure in figures folder
    plt.savefig(f"figures/{name}_tMC.png")
    plt.close()

def plot_fss(T_s, results, L_s, observable, ylabel, title):

    fig, ax = plt.subplots(figsize=(8, 5))
    for L in L_s:
        y    = results[L][observable]
        yerr = results[L][observable + '_err']
        ax.errorbar(T_s, y, yerr=yerr, label=f'L={L}', marker='o', capsize=3)

    ax.axvline(x=2.269, color='red', linestyle='--', label='$T_c$ (esatto)')
    ax.set_xlabel('T')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    #grid
    ax.grid()
    fig.savefig(f'figures/fss_{observable}.png', dpi=150)
    plt.close(fig)

def config_plot(configs, T_s):
    #plot of 3 configuration of the lattice at different temperatures: ordered, critical and disordered
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    #titles with the corresponding temperatures
    titles = [f"Ordered (T={T_s[0]:.2f})", f"Near Critical (T={T_s[len(T_s) // 2]:.2f})", f"Disordered (T={T_s[-1]:.2f})"]
    for ax, config, title in zip(axes, configs, titles):
        #plot the configuration in black and white
        im = ax.imshow(config, cmap='gray', vmin=-1, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig('figures/configurations.png', dpi=150)
    plt.close(fig)
