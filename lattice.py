import random

def create_lattice (L, initial_state='random'):
    if initial_state == 'random':
        return [[random.choice([-1,1]) for _ in range(L)]for _ in range(L)]
    elif initial_state == 'up':
        return [[1 for _ in range(L)]for _ in range(L)]
    elif initial_state == 'down':
        return [[-1 for _ in range(L)]for _ in range(L)]
    else:
        print("Invalid initial state. Choose 'random', 'up', or 'down'.")

def flip_random_spin(lattice):
    i = random.randrange(len(lattice))
    j = random.randrange(len(lattice))
    lattice[i][j] *= -1
    return (i, j)

def print_lattice(lattice):
    for row in lattice:
        print(row)