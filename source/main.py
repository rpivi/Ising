import numpy as np
import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro
import pca as pca

def main():

# Part1: Thermalization analysis of different initial states at fixed BJ
    L = 50  # Lattice size
    N = L * L  # Total number of spins
    T = 1.4
    BJ = 1/T    # Inverse temperature
    nsweep_therm = 1000 # Number of sweeps for thermalization 
    sweep_skip_therm = 1
    np.random.seed(42)

    spins_r = lat.create_lattice(L, initial_state='random')
    spins_u = lat.create_lattice(L, initial_state='up')

    _, magn_r, tot_energies_r, spins_r = metro.metropolis(spins_r, nsweep_therm, sweep_skip_therm, BJ)
    _, magn_u, tot_energies_u, spins_u= metro.metropolis(spins_u, nsweep_therm, sweep_skip_therm, BJ)

    plot.plot_two_steps(magn_r/N, magn_u/N, name="Mean Magnetization", name1="Random Init", name2="Up Init", BJ=BJ)
    plot.plot_two_steps(tot_energies_r/N, tot_energies_u/N, name="Mean Energy", name1="Random Init", name2="Up Init", BJ=BJ)
    print("Thermalization analysis completed and plots saved.")

    # Part2: Phase transition analysis + finite-size scaling + T_c estimation
    L_s = [30,50]  # Lattice sizes
    T_s = np.concatenate([
    np.linspace(1.7, 2.2, 5, endpoint=False),   # ordered phase: T < T_c:
    np.linspace(2.2,  2.3,  5, endpoint=False),   # critical region: T ~ T_c
    np.linspace(2.3, 2.8, 6) ])    # disordered phase: T > T_c 
    T_s = np.sort(np.append(T_s, 2.269)) # Ensure T_c is included in the temperature list
    
    BJ_s = [1/T for T in T_s]

    nsamples = 300
    sweeps_skip = 50 # Number of sweeps between samples to ensure decorrelation
    nsweep_tot = nsamples * sweeps_skip # Total number of sweeps to get nsamples independent measurements

    # Dictionary for storing results for each lattice size L
    results = {L: {
        'mean_magnetizations':    [],
        'mean_energies':          [],
        'heat_capacity':          [],
        'susceptibility':         [],
        'mean_magnetizations_err':[],
        'mean_energies_err':      [],
        'heat_capacity_err':      [],
        'susceptibility_err':     [],
        'spins_configs':          [],
    } for L in L_s}

    for L in L_s:
        N = L * L
        spins = lat.create_lattice(L, initial_state='random')
        for T, BJ in zip(T_s, BJ_s):
            # Thermalization
            nsweep_therm_adaptive = metro.get_therm_sweeps(T) #500 sweeps for T far from T_c, 1000 sweeps for T close to T_c
            _, _, _, spins = metro.metropolis(spins, nsweep_therm_adaptive, sweep_skip_therm, BJ)

            # Measurement of observables
            net_spins, abs_magnet, net_energies, _, configs = metro.metropolis(
                spins, nsweep_tot, sweeps_skip, BJ, return_configs=True
            )

            # Observables
            m  = abs_magnet          # |M|
            E  = net_energies
            mean_m  = np.mean(m)

            results[L]['mean_magnetizations'].append(mean_m / N)
            results[L]['mean_energies'].append(np.mean(E) / N)
            results[L]['heat_capacity'].append(ph.heat_capacity(E, BJ, N))
            results[L]['susceptibility'].append(ph.susceptibility(net_spins, BJ, N))

            # Assumption of inidipendent samples: error on mean via standard deviation / sqrt(nsamples)
            results[L]['mean_magnetizations_err'].append(np.std(m / N) / np.sqrt(nsamples))
            results[L]['mean_energies_err'].append(np.std(E / N) / np.sqrt(nsamples))

            energy_variance_err = np.std((E - np.mean(E))**2, ddof=1) / np.sqrt(nsamples)
            results[L]['heat_capacity_err'].append((BJ**2 / N) * energy_variance_err)

            spin_variance_err = np.std((net_spins - np.mean(net_spins))**2, ddof=1) / np.sqrt(nsamples)
            results[L]['susceptibility_err'].append((BJ / N) * spin_variance_err)

            results[L]['spins_configs'].append(configs)
            print(f"  L={L}, T={T:.2f} completato")
            spins = configs[-1]  # Use the last configuration as the starting point for the next temperature

        print(f"Completate misure per L={L}")

    # Plot finite-size scaling
    plot.plot_fss(T_s, results, L_s, observable='mean_magnetizations',
                  ylabel='|M|/N', title='Magnetizzazione media')
    plot.plot_fss(T_s, results, L_s, observable='mean_energies',
                  ylabel='E/N', title='Energia media')
    plot.plot_fss(T_s, results, L_s, observable='heat_capacity',
                  ylabel='Cv', title='Capacità termica')
    plot.plot_fss(T_s, results, L_s, observable='susceptibility',
                  ylabel='χ', title='Suscettività')

    # T_c from Chi and Cv peak estimation
    T_c_estimates = []
    T_c_estimates_cv = []
    for L in L_s:
        T_c, T_c_err = ph.find_T_peak(T_s, results[L]['susceptibility'])
        T_c_estimates.append((T_c, T_c_err))
        print(f"Estimated T_c from χ for L={L}: {T_c:.3f} ± {T_c_err:.3f}")
        T_c_cv, T_c_cv_err = ph.find_T_peak(T_s, results[L]['heat_capacity'])
        T_c_estimates_cv.append((T_c_cv, T_c_cv_err))
        print(f"Estimated T_c from Cv for L={L}: {T_c_cv:.3f} ± {T_c_cv_err:.3f}")

    #plotting of 3 configuration of the lattice at different temperatures: ordered, critical and disordered for L=50
    #taking the last configuration for each temperature as representative
    L_plot = 50
    config_ordered = results[L_plot]['spins_configs'][0][-1]  # First temperature (ordered phase)
    config_critical = results[L_plot]['spins_configs'][len(T_s) // 2][-1]  # Middle temperature (critical region)
    config_disordered = results[L_plot]['spins_configs'][-1][-1]  # Last temperature (disordered phase)
    configs_to_plot = [config_ordered, config_critical, config_disordered]
    plot.config_plot(configs_to_plot, T_s)
    
    # Part3: PCA analysis of spin configurations at L=50
    L_pca = 50
    all_configs, T_labels = pca.prepare_pca_data(results[L_pca]['spins_configs'], T_s)
    X_pca, explained_var_ratio = pca.perform_pca(all_configs, n_components=2)
    pca.pca_plot(X_pca, T_labels, explained_var_ratio)
    mean_abs_fpc = pca.pca_fpc_T(X_pca, T_labels)
    mean_bas_spc = pca.pca_spc_T(X_pca, T_labels)
    corr_pc1_m = np.corrcoef(mean_abs_fpc, results[L_pca]['mean_magnetizations'])[0,1]
    print(f"Correlation between PC1 and magnetization: {corr_pc1_m:.3f}")
    corr_pc2_chi= np.corrcoef(mean_bas_spc, results[L_pca]['susceptibility'])[0,1]
    print(f"Correlation between PC2 and susceptibility: {corr_pc2_chi:.3f}")
    print("PCA analysis completed and plots saved.")
if __name__ == "__main__":
    main()