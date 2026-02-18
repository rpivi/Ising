import numpy as np
import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro
import pca as pca

def main():

# Part1: Thermalization analysis of different initial states at fixed BJ
    L = 60  # Lattice size
    N = L * L  # Total number of spins
    T = 1.4
    BJ = 1/T    # Inverse temperature
    nsweep_therm = {20: 500, 40: 1000, 60: 1500} # Number of sweeps for thermalization for each L
    sweep_skip_therm = 1
    np.random.seed(42)

    spins_r = lat.create_lattice(L, initial_state='random')
    spins_u = lat.create_lattice(L, initial_state='up')

    _, magn_r, tot_energies_r, spins_r = metro.metropolis(spins_r, nsweep_therm[L], sweep_skip_therm, BJ)
    _, magn_u, tot_energies_u, spins_u= metro.metropolis(spins_u, nsweep_therm[L], sweep_skip_therm, BJ)

    plot.plot_two_steps(magn_r/N, magn_u/N, name="Mean Magnetization", name1="Random Init", name2="Up Init", BJ=BJ)
    plot.plot_two_steps(tot_energies_r/N, tot_energies_u/N, name="Mean Energy", name1="Random Init", name2="Up Init", BJ=BJ)
    print("Thermalization analysis completed and plots saved.")

    # Part2: Phase transition analysis + finite-size scaling + T_c estimation
    L_s = [20, 40, 60]  # Lattice sizes
    T_s = np.concatenate([
    np.linspace(1.5,  2.2,  7, endpoint=False),   # ordered phase: T < T_c
    np.linspace(2.2,  2.3,  10, endpoint=False),   # critical region: T ~ T_c
    np.linspace(2.3, 3.0, 8) ])    # disordered phase: T > T_c 
    
    BJ_s = [1/T for T in T_s]

    nsamples = 200
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
        for T, BJ in zip(T_s, BJ_s):
            # Thermalization
            spins = lat.create_lattice(L, initial_state='random')
            _, _, _, spins = metro.metropolis(spins, nsweep_therm[L], sweep_skip_therm, BJ)

            # Misure
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
    
    # Estimate T_c from the peaks of susceptibility for each L
    T_peaks     = []
    T_peaks_err = []

    for L in L_s:
        T_peak, T_peak_err = ph.find_susceptibility_peak(T_s, results[L]['susceptibility'])
        T_peaks.append(T_peak)
        T_peaks_err.append(T_peak_err)

    Tc_fsc, Tc_err = plot.finite_size_scaling_Tc(L_s, T_peaks, T_peaks_err)
    print(f"Estimated T_c from finite-size scaling: {Tc_fsc:.4f} ± {Tc_err:.4f}")

    # Part3: PCA analysis of spin configurations at L=20
    L_pca = 20
    all_configs, T_labels = pca.prepare_pca_data(results[L_pca]['spins_configs'], T_s)
    X_pca, explained_var_ratio = pca.perform_pca(all_configs, n_components=2)
    pca.pca_plot(X_pca, T_labels, explained_var_ratio)
    pca.pca_fpc_T(all_configs, T_labels)
    pca.pca_spc_T(all_configs, T_labels)
    print("PCA analysis completed and plots saved.")

if __name__ == "__main__":
    main()