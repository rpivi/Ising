import numpy as np

def create_lattice(L, initial_state='random'):
        init_random = np.random.random((L, L))
        spins = np.zeros((L, L), dtype=int)
        if initial_state == 'random':
            spins [init_random < 0.5] = -1
            spins [init_random >= 0.5] = 1
        elif initial_state == 'up':
            spins [init_random < 0.80] = 1
            spins [init_random >= 0.80] = -1
        elif initial_state == 'down':
            spins [init_random < 0.80] = -1
            spins [init_random >= 0.80] = 1
        else:
            print("Error! initial_state must be 'random', 'up', or 'down'")
        return spins