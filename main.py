import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro
import numpy as np

def main():

    L = 50  # Lattice size
    N = L * L  # Total number of spins
    BJ = 0.7  # Inverse temperature
    steps = 100000 +1 # Number of Monte Carlo steps

    spins_r = lat.create_lattice(L, initial_state='random')
    spins_u = lat.create_lattice(L, initial_state='up')

    initial_energy_r = ph.total_energy(spins_r)
    initial_energy_u = ph.total_energy(spins_u)

    net_spins_r, net_energies_r = metro.metropolis(spins_r, steps, BJ, initial_energy_r)
    net_spins_u, net_energies_u = metro.metropolis(spins_u, steps, BJ, initial_energy_u)

    plot.plot_two_steps(net_spins_r/N, net_spins_u/N, name1="Mean Spins (Random Init)", name2="Mean Spins (Up Init)", BJ=BJ)
    plot.plot_two_steps(net_energies_r, net_energies_u, name1="Net Energy (Random Init)", name2="Net Energy (Up Init)", BJ=BJ)

if __name__ == "__main__":
    main()