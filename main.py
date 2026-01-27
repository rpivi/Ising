import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro
import numpy as np

def main():

# Part1: Thermalization analysis of different initial states at fixed BJ ########
    L = 50  # Lattice size
    N = L * L  # Total number of spins
    BJ = 0.7  # Inverse temperature
    nsweep_therm = 700  # Number of sweeps
    np.random.seed(42)

    spins_r = lat.create_lattice(L, initial_state='random')
    spins_u = lat.create_lattice(L, initial_state='up')

    magn_r, tot_energies_r, spins_r = metro.metropolis(spins_r, nsweep_therm, BJ)
    magn_u, tot_energies_u, spins_u= metro.metropolis(spins_u, nsweep_therm, BJ)

    plot.plot_two_steps(magn_r/N, magn_u/N, name="Mean Magnetization", name1="Random Init", name2="Up Init", BJ=BJ)
    plot.plot_two_steps(tot_energies_r, tot_energies_u, name="Total Energy", name1="Random Init", name2="Up Init", BJ=BJ)
    print("Thermalization analysis completed and plots saved.")

# Part2: Phase transition analysis  #############################################
    BJ_s = [0.1,0.2,0.3,0.4,0.41,0.44,0.46,0.5,0.6,0.7,0.8,1.0,1.5,2.0, 2.5, 3.0]

    mean_magnetizations = []
    mean_energies = []
    heat_capacity = []
    susceptibility = []

     # Loop over different BJ values
    for BJ in BJ_s:
        #thermalization
        spins = lat.create_lattice(L, initial_state='random')
        _,_,spins = metro.metropolis(spins, nsweep_therm, BJ)

        #measurement of mean magnetization, mean energy, heat capacity and susceptibility
        nsweep_meas = 500 # Number of sweeps for measurement = number of samples at each BJ
        net_spins, net_energies, _ = metro.metropolis(spins, nsweep_meas, BJ)
        mean_magnetizations.append(np.mean(net_spins)/N)
        mean_energies.append(np.mean(net_energies)/N)
        heat_capacity.append(ph.heat_capacity(net_energies, BJ, N))
        susceptibility.append(ph.susceptibility(net_spins, BJ, N))
        print(f"Completed measurements at BJ={BJ}")

    # Plot observables vs BJ
    plot.plot_vs_BJ(BJ_s, mean_magnetizations, name="Mean Magnetization")
    plot.plot_vs_BJ(BJ_s, mean_energies, name="Mean Energy")
    plot.plot_vs_BJ(BJ_s, heat_capacity, name="Heat Capacity")
    plot.plot_vs_BJ(BJ_s, susceptibility, name="Susceptibility")

if __name__ == "__main__":
    main()