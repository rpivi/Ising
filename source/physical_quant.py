import lattice as lat
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure
from numba import njit

@njit(cache=True)
def total_energy(spins):
    L = spins.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            s = spins[i, j]
            E -= s * (spins[(i + 1) % L, j] + spins[i, (j + 1) % L])
    return E

def heat_capacity(energies, T, N):
    """
    Calore specifico per sito: Cv = Var(E) / (T^2 * N)
    Input: energies = array di energie totali (non per sito)
           T  = temperatura (non BJ)
           N  = numero di siti = L*L
    """
    energies = np.asarray(energies, dtype=np.float64)
    return np.var(energies) / (T**2 * N)

def susceptibility(magnetizations, T, N):
    """
    Suscettività per sito: chi = Var(|M|) / (T * N)
    Input: magnetizations = array di |M| totali (non per sito)
           T  = temperatura
           N  = numero di siti
    """
    magnetizations = np.asarray(magnetizations, dtype=np.float64)
    return np.var(magnetizations) / (T * N)

def find_T_peak(T_s, observable):
    T_arr   = np.array(T_s)
    obs_arr = np.array(observable)

    idx_peak = np.argmax(obs_arr)
    T_peak   = T_arr[idx_peak]

    # Estimate the error on T_peak using the width of the peak at half maximum
    half_max = obs_arr[idx_peak] / 2
    # Find the indices where the susceptibility is greater than half maximum
    indices_half_max = np.where(obs_arr >= half_max)[0]
    if len(indices_half_max) > 1:
        T_left = T_arr[indices_half_max[0]]
        T_right = T_arr[indices_half_max[-1]]
        T_peak_err = (T_right - T_left) / 2
    else:
        T_peak_err = 0.0

    return T_peak, T_peak_err

