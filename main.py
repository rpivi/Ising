import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro
import numpy as np

def main():

    L = 20        # lattice size
    steps = 4000  #number of Metropolis steps
    N = L * L   # number of spins
    T = 0.2      # temperature in units of J/kb
    np.random.seed(42)  # for reproducibility

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
        if step % 100 == 0:
            print(f"Step {step}/{steps} completed.")
            mean_energy_rand.append(E_rand / N)
            mean_magnet_rand.append(M_rand/ N)
            mean_energy_up.append(E_up / N)
            mean_magnet_up.append(M_up/ N) 

    print("Metropolis algorithm completed.")

    # Plotting the results 
    plot.plot_two_steps(mean_energy_rand, mean_energy_up, name1="Mean Energy (Random Init)", name2="Mean Energy (All Up Init)", T=T)
    plot.plot_two_steps(mean_magnet_rand, mean_magnet_up, name1="Mean Magnetization (Random Init)", name2="Mean Magnetization (All Up Init)", T=T)
    print("Plots saved.")

    # # Now we now that its termalaized we can run a longer simulation to compute observables at equilibrium
    # print("Running simulation for equilibrium observables...")

    # # Re-initialize lattice it has already thermalized
    # steps = 10e4
    # lat_eq = lat_rand
    # E_eq = ph.total_energy(lat_eq)
    # M_eq = ph.magnetization(lat_eq)
    # energies = []
    # magnetizations = [] 

    # for step in range(steps):
    #     for i in range(N):
    #         dE_eq, dM_eq = metro.metropolis_step(lat_eq, T)
    #         E_eq += dE_eq
    #         M_eq += dM_eq
    #     if step % 50 == 0:
    #       print(f"Equilibrium Step {step}/{steps} completed.")
    #       energies.append(E_eq)
    #       magnetizations.append(M_eq)
    # print("Equilibrium simulation completed.")
    #observables
    # mean_E = np.mean(np.array(energies)) / N
    # mean_M = np.mean(np.array(magnetizations)) / N

    #plotting
    # plot.plot_step(mean_E, name="Mean Energy at Equilibrium", T=T)
    # plot.plot_step(mean_M, name="Mean Magnetization at Equilibrium", T=T)
    # print("Equilibrium plots saved.")

if __name__ == "__main__":
    main()