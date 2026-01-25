import numpy as np

def create_lattice(L, initial_state='random'):
    if initial_state == 'random':
        init_random = np.random.rand(L,L)
        lattice = np.where(init_random < 0.5, -1, 1)
        return lattice
    elif initial_state == 'up':
        return np.ones((L, L), dtype=int)
    elif initial_state == 'down':
        return -np.ones((L, L), dtype=int)
    else:
        raise ValueError("Choose 'random', 'up', or 'down'")

def print_lattice(lattice):
    for row in lattice:
        print(" ".join(f"{s:2d}" for s in row))