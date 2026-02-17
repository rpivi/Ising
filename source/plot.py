import matplotlib.pyplot as plt
import physical_quant as ph
import numpy as np  


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

def plot_vs_T_errors(T_s, obs_s, obs_err_s, name="Observable"):
    plt.figure(figsize=(8,6))
    plt.errorbar(T_s, obs_s, yerr=obs_err_s, fmt='o', capsize=5)
    plt.grid()
    plt.title(f"{name} vs T")
    plt.xlabel("T")
    plt.ylabel(name)
    if name == "Mean Magnetization":
        plt.ylim(0, 1)
        # Soluzione esatta di Yang
        T_exact = np.linspace(min(T_s) * 0.9, max(T_s) * 1.1, 500)
        Tc = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269 con J=kB=1
        M_exact = np.zeros_like(T_exact)
        mask = T_exact < Tc
        M_exact[mask] = (1 - np.sinh(2.0 / T_exact[mask])**(-4))**(1/8)
        plt.plot(T_exact, M_exact, 'k-', lw=2, label="Yang (1952) exact")
    #line in the Tc
    plt.axvline(x=2.269, color='red', linestyle='--', label='Critical Temperature Tc')
    plt.legend()
    #save figure in figures folder
    plt.savefig(f"figures/{name}_vs_T_errors.png")
    plt.close()

def spatial_correlation_plot(spins_configs, T_s, L):
    T_plot = [1.4, 2.27, 3.0] # ordered, critical and disordered phases
    
    plt.figure(figsize=(8, 5))
    
    for T_target in T_plot:
        idx = T_s.index(T_target)
        r_values, C_r = ph.mean_spatial_correlation(spins_configs[idx], L)
        plt.plot(r_values, C_r, label=f'T={T_target}', marker='o', markersize=3)
    
    plt.xlabel('r')
    plt.ylabel('C(r)')
    plt.title('Spatial Correlation Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/spatial_correlation.png', dpi=150)
    plt.close()