import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro

def main():

    L = 50        # lattice size
    steps = 7000  #number of Metropolis steps
    N = L * L   # number of spins
    T = 4.5      # temperature

    #visualize the termalization
    lat_rand = lat.create_lattice(L, initial_state='random')
    lat_up =  lat.create_lattice(L, initial_state='up')
    print("Lattice created")

    # initial values of E and M for the 2 configurations
    E_rand = ph.total_energy(lat_rand)
    M_rand = ph.magnetization(lat_rand)
    E_up = ph.total_energy(lat_up)
    M_up = ph.magnetization(lat_up)

    mean_energy_rand = []
    mean_magnet_rand = []
    mean_energy_up = []
    mean_magnet_up = []

    print("Starting Metropolis algorithm...")

    for step in range(steps):
        for i in range(N):
            dE_rand, dM_rand = metro.metropolis_step(lat_rand, T)
            E_rand += dE_rand
            M_rand += dM_rand

            dE_up, dM_up = metro.metropolis_step(lat_up, T)
            E_up += dE_up
            M_up += dM_up

        mean_energy_rand.append(E_rand / N)
        mean_magnet_rand.append(M_rand / N)
        mean_energy_up.append(E_up / N)
        mean_magnet_up.append(M_up / N) 

    print("Metropolis algorithm completed.")

    # Plotting the results 
    plot.plot_two_steps(mean_energy_rand, mean_energy_up, name1="Mean Energy (Random Init)", name2="Mean Energy (All Up Init)", T=T)
    plot.plot_two_steps(mean_magnet_rand, mean_magnet_up, name1="Mean Magnetization (Random Init)", name2="Mean Magnetization (All Up Init)", T=T)
if __name__ == "__main__":
    main()