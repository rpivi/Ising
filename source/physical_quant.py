import lattice as lat
import numpy as np
from scipy.ndimage import convolve, generate_binary_structure

def total_energy(spins):
    kern = generate_binary_structure(2,1)
    kern[1,1] = False
    arr = - spins * convolve(spins, kern, mode='wrap', cval=0) #periodic boundary conditions
    return arr.sum()/2

def heat_capacity(energies, BJ, N):
    E2_mean = np.mean(np.array(energies)**2)
    E_mean = np.mean(energies)
    C = (BJ**2 / N) * (E2_mean - E_mean**2)
    return C

def susceptibility(magnetizations, BJ, N):
    M2_mean = np.mean(np.array(magnetizations)**2)
    M_mean = np.mean(magnetizations)
    chi = (BJ / N) * (M2_mean - M_mean**2)
    return chi

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

