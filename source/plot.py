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

    ax.axvline(x=2.269, color='gray', linestyle='--', label='$T_c$ (esatto)')
    ax.set_xlabel('T')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    #grid
    ax.grid()
    fig.savefig(f'figures/fss_{observable}.png', dpi=150)
    plt.close(fig)


def finite_size_scaling_Tc(L_s, T_peaks, T_peaks_err):
    nu_ising_2d=1.0
    L_arr = np.array(L_s, dtype=float)
    T_arr = np.array(T_peaks)
    err_arr = np.array(T_peaks_err)

    def model(L, Tc, a):
        return Tc + a * L**(-1.0 / nu_ising_2d)

    popt, pcov = curve_fit(model, L_arr, T_arr, sigma=err_arr,
                           absolute_sigma=True, p0=[2.269, 0.5])
    Tc_fit, a_fit = popt
    Tc_err = np.sqrt(pcov[0, 0])

    L_fine = np.linspace(L_arr.min() * 0.8, L_arr.max() * 1.5, 500)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(L_arr, T_arr, yerr=err_arr, fmt='o', color='black',
                label='$T_{peak}(L)$', capsize=4, zorder=5)
    ax.plot(L_fine, model(L_fine, *popt), '-',
            label=f'fit ν={nu_ising_2d}\n$T_c = {Tc_fit:.4f} \\pm {Tc_err:.4f}$')
    ax.axhline(2.2692, color='gray', linestyle=':', label='$T_c$ esatto = 2.2692')
    ax.set_xlabel('L')
    ax.set_ylabel('$T_{peak}(L)$')
    ax.set_title('Finite-Size Scaling — picchi della suscettività')
    ax.legend()
    #grid
    ax.grid()
    fig.tight_layout()
    fig.savefig('figures/fss_Tc.png', dpi=150)
    plt.close(fig)
    return Tc_fit, Tc_err