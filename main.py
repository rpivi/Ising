import utils as ut
import lattice as lat
import observables as obs
import plot as pl
import metropolis as metro

def main():

    N = 20        # dimensione griglia
    T = 2.0       # temperatura
    J = 1.0       # costante di accoppiamento
    
    print("Dimensione:", N)
    print("Temperatura:", T)

    lattice_state = lat.create_lattice(N, initial_state='random')
    lat.print_lattice(lattice_state)

    coord = lat.flip_random_spin(lattice_state)
    print(f"\nAfter flipping a random spin in {coord} :\n")
    lat.print_lattice(lattice_state)

if __name__ == "__main__":
    main()