import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro
import numpy as np
import pca as pca

def main():

# Part1: Thermalization analysis of different initial states at fixed BJ
    L = 50  # Lattice size
    N = L * L  # Total number of spins
    BJ = 0.7  # Inverse temperature
    nsweep_therm = 700  # Number of sweeps
    sweep_skip_therm = 1
    np.random.seed(42)

    spins_r = lat.create_lattice(L, initial_state='random')
    spins_u = lat.create_lattice(L, initial_state='up')

    magn_r, tot_energies_r, spins_r = metro.metropolis(spins_r, nsweep_therm, sweep_skip_therm, BJ)
    magn_u, tot_energies_u, spins_u= metro.metropolis(spins_u, nsweep_therm, sweep_skip_therm, BJ)

    plot.plot_two_steps(magn_r/N, magn_u/N, name="Mean Magnetization", name1="Random Init", name2="Up Init", BJ=BJ)
    plot.plot_two_steps(tot_energies_r, tot_energies_u, name="Total Energy", name1="Random Init", name2="Up Init", BJ=BJ)
    print("Thermalization analysis completed and plots saved.")

# Part2: Phase transition analysis 
    T_s = [1.5,2.,2.1,2.2,2.25,2.26,2.27,2.28,2.29,2.3,2.4,2.5,3]
    BJ_s =[ 1/T for T in T_s]
    spins_configs= []

    mean_magnetizations = []
    mean_energies = []
    heat_capacity = []
    susceptibility = []

     # Loop over different BJ values
    for BJ in BJ_s:
        #thermalization
        spins = lat.create_lattice(L, initial_state='random')
        _,_,spins = metro.metropolis(spins, nsweep_therm,sweep_skip_therm, BJ)

        #measurement of mean magnetization, mean energy, heat capacity and susceptibility
        nsamples = 200
        sweeps_skip = 50
        nsweep_meas = nsamples * sweeps_skip
        net_spins, net_energies, _, spins_configs = metro.metropolis(spins, nsweep_meas,sweeps_skip,BJ, return_configs=True)
        mean_magnetizations.append(np.mean(net_spins)/N)
        mean_energies.append(np.mean(net_energies)/N)
        heat_capacity.append(ph.heat_capacity(net_energies, BJ, N))
        susceptibility.append(ph.susceptibility(net_spins, BJ, N))
        spins_configs.append(spins) # Store the final spin configuration for PCA analysis
        print(f"Completed measurements at BJ={BJ}")

    # Plot observables vs BJ
    plot.plot_vs_BJ(BJ_s, mean_magnetizations, name="Mean Magnetization")
    plot.plot_vs_BJ(BJ_s, mean_energies, name="Mean Energy")
    plot.plot_vs_BJ(BJ_s, heat_capacity, name="Heat Capacity")
    plot.plot_vs_BJ(BJ_s, susceptibility, name="Susceptibility")

# Part3: PCA analysis of spin configurations at different BJ values
    X_pca, explained_variance_ratio = pca.perform_pca(spins_configs, n_components=2)
    pca.pca_plot(X_pca, BJ_s)
    print("PCA analysis completed.")

if __name__ == "__main__":
    main()