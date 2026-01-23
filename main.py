import utils as ut
import lattice as lat
import physical_quant as ph
import plot as plot
import metropolis as metro

def main():

    L = 50        # dimensione griglia
    T = 1      # temperatura, kb = 1
    J = 1    # costante di accoppiamento
    steps = 500000  #steps di Metropolis
    N = L * L   # numero di spin

    lattice_state = lat.create_lattice(L, initial_state='random')

    # misure
    energies_J = []
    mags_N= []

    for _ in range(steps):
        metro.metropolis_step(lattice_state, T)
        energies_J.append(ph.total_energy(lattice_state, J)/J)
        mags_N.append(ph.magnetization(lattice_state)/N)

    plot.plot_step(energies_J, name="Energy/J")
    plot.plot_step(mags_N, name="Magnetization/N")

if __name__ == "__main__":
    main()

#modificare in modo che il progetto dipenda da beta*J = J/(kB*T) come unico parametro
#modificare in modo che il progetto calcoli anche la suscettività magnetica e la capacità termica
#energia direttamente E/J
#algoritmo addizionale per energia e magnetizzazione: non il calcolo della tot tutti i passsi ma la variazione ad ogni step di Metropolis
