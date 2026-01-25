import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro

def main():

    L = 50        # lattice size
    BJ = [0.2, 0.4, 0.44, 0.6] # 1/(kB*T) with kB=1
    steps = 500000  #number of Metropolis steps
    N = L * L   # number of spins

    for bj in BJ:
        lattice_state = lat.create_lattice(L, initial_state='random')
    
        # observables
        energy = []
        mean_energy = []
        magnetization= [] 
        mean_magnet = []

        # initial values of E and M
        E = ph.total_energy(lattice_state)
        M = ph.magnetization(lattice_state)

        for _ in range(steps):
            dE, dM = metro.metropolis_step(lattice_state, bj)
            E += dE
            M += dM
            energy.append(E)
            mean_energy.append(E/N)
            magnetization.append(M)
            mean_magnet.append(M/N)

           
        plot.plot_step(mean_energy, name="Mean Energy [in units of J]", BJ=bj)
        plot.plot_step(mean_magnet, name="Mean Magnetization", BJ=bj)

if __name__ == "__main__":
    main()