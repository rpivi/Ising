import numpy as np
import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro
import pca as pca
import statistics as stats

def main():

# Part1: Thermalization analysis of different initial states at fixed BJ
    L = 20  # Lattice size
    N = L * L  # Total number of spins
    BJ = 0.7  # Inverse temperature
    nsweep_therm = 500  # Number of sweeps
    sweep_skip_therm = 1
    np.random.seed(42)

    spins_r = lat.create_lattice(L, initial_state='random')
    spins_u = lat.create_lattice(L, initial_state='up')

    _, magn_r, tot_energies_r, spins_r = metro.metropolis(spins_r, nsweep_therm, sweep_skip_therm, BJ)
    _, magn_u, tot_energies_u, spins_u= metro.metropolis(spins_u, nsweep_therm, sweep_skip_therm, BJ)

    plot.plot_two_steps(magn_r/N, magn_u/N, name="Mean Magnetization", name1="Random Init", name2="Up Init", BJ=BJ)
    plot.plot_two_steps(tot_energies_r, tot_energies_u, name="Total Energy", name1="Random Init", name2="Up Init", BJ=BJ)
    print("Thermalization analysis completed and plots saved.")

# Part2: Phase transition analysis 
    T_s = [1.5,2.,2.2,2.25,2.26,2.27,2.28,2.29,2.3,2.4,2.5,3]
    BJ_s =[ 1/T for T in T_s]
    spins_configs= []

    mean_magnetizations = []
    mean_energies = []
    heat_capacity = []
    susceptibility = []

    mean_magnetizations_err = []
    mean_energies_err = []  
    heat_capacity_err = []
    susceptibility_err = []

     # Loop over different BJ values
    for BJ in BJ_s:
        #thermalization
        spins = lat.create_lattice(L, initial_state='random')
        _, _, _,spins = metro.metropolis(spins, nsweep_therm,sweep_skip_therm, BJ)

        #measurement of mean magnetization, mean energy, heat capacity and susceptibility
        nsamples = 200
        sweeps_skip = 50
        nsweep_meas = nsamples * sweeps_skip
        net_spins, abs_magnet, net_energies, _, configs = metro.metropolis(spins, nsweep_meas,sweeps_skip, BJ, return_configs=True)

        #calculating observables
        mean_magnetizations.append(np.mean(abs_magnet)/N)
        mean_energies.append(np.mean(net_energies)/N)
        heat_capacity.append(ph.heat_capacity(net_energies, BJ, N))
        susceptibility.append(ph.susceptibility(net_spins, BJ, N))

        #calculating errors using standard error of the mean: std/sqrt(n)
        mean_magnetizations_err.append(stats.sem(abs_magnet/N))
        mean_energies_err.append(stats.sem(net_energies/N))

        # For heat capacity and susceptibility with error propagation
        heat_capacity_err.append(stats.sem(net_energies**2/N**2)*(BJ**2))
        susceptibility_err.append(stats.sem(net_spins**2/N**2)*(BJ))
        
        #saving configurations for PCA 
        spins_configs.append(configs)
        print(f"Completed measurements at BJ={BJ}")

    # Plotting observables vs T
    plot.plot_vs_T_errors(T_s, mean_magnetizations,mean_energies_err, name="Mean Magnetization")
    plot.plot_vs_T_errors(T_s, mean_energies, mean_energies_err, name="Mean Energy")
    plot.plot_vs_T_errors(T_s, heat_capacity, heat_capacity_err, name="Heat Capacity")
    plot.plot_vs_T_errors(T_s, susceptibility, susceptibility_err, name="Susceptibility")

# Part3: PCA analysis of spin configurations at different BJ values
    all_configs, T_labels = pca.prepare_pca_data(spins_configs, T_s)
    X_pca , explained_var_ratio= pca.perform_pca(all_configs, n_components=2)
    pca.pca_plot(X_pca, T_labels, explained_var_ratio)

if __name__ == "__main__":
    main()