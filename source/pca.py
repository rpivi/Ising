from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np

def prepare_pca_data(spins_configs_nested, T_s):
    all_configs = []
    T_labels = []
    
    for T, configs in zip(T_s, spins_configs_nested):
        all_configs.extend(configs)
        T_labels.extend([T] * len(configs))
    
    return all_configs, T_labels


def perform_pca(configs_flat, n_components=2):
    # Appiattisci ogni configurazione 2D in vettore 1D
    X = np.array([config.flatten() for config in configs_flat])
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca.explained_variance_ratio_

def pca_plot(X_pca, T_labels, explained_variance_ratio=None):
    T_labels = np.array(T_labels)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=T_labels, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, label='Temperature T')

    xlabel = "Principal Component 1"
    ylabel = "Principal Component 2"
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title("PCA of Spin Configurations - Ising Model", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #save in figures folder
    plt.savefig("figures/PCA_spin_configurations.png", dpi=300)
    plt.close()

def pca_fpc_T(X, T_labels):
    # study the first principal component as a function of T

    X = np.asarray(X)
    T_labels = np.asarray(T_labels)

    T_unique = np.unique(T_labels)
    mean_fpc = []

    for T in T_unique:
        mean_fpc.append(X[T_labels == T, 0].mean())

    plt.figure(figsize=(8,6))
    plt.plot(T_unique, mean_fpc, marker='o')
    plt.xlabel("Temperature T")
    plt.ylabel("Mean First Principal Component")
    plt.title("Mean First Principal Component vs Temperature")
    plt.grid()
    plt.savefig("figures/Mean_FPC_vs_T.png", dpi=300)
    plt.close()

