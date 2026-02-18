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

def pca_plot(X_pca, T_labels, explained_var_ratio):
    T_labels = np.array(T_labels)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=T_labels, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, label='Temperature T')
    
    xlabel = f"Principal Component 1 ({explained_var_ratio[0]*100:.1f}% var)"
    ylabel = f"Principal Component 2 ({explained_var_ratio[1]*100:.1f}% var)"
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title("PCA of Spin Configurations - Ising Model", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #save in figures folder
    plt.savefig("figures/PCA_spin_configurations.png", dpi=300)
    plt.close()

def pca_fpc_T(configs_flat, T_labels):
    # study the mean absolute value of the first principal component as a function of T

    X = np.array([config.flatten() for config in configs_flat])
    T_labels = np.array(T_labels)

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X)

    mean_abs_fpc = np.array([np.mean(np.abs(X_pca[T_labels == T])) for T in np.unique(T_labels)])

    #plotting the mean absolute value of the firste component vs T
    plt.figure(figsize=(10, 6))
    plt.scatter(T_labels, mean_abs_fpc, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    #line in the Tc
    plt.axvline(x=2.269, color='red', linestyle='--', label='Critical Temperature Tc')
    plt.legend()
    plt.xlabel('Temperature T', fontsize=12)
    plt.ylabel('Mean Absolute Value of the First PC', fontsize=12)
    plt.title('Mean Absolute First PC vs Temperature', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #save in figures folder
    plt.savefig("figures/PCA_first_component_vs_T.png", dpi=300)
    plt.close()

def pca_spc_T(configs_flat, T_labels):
    # study the mean absolute value of the second principal component vs T

    X = np.array([config.flatten() for config in configs_flat])
    T_labels = np.array(T_labels)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    mean_abs_spc = np.array([np.mean(np.abs(X_pca[T_labels == T, 1])) for T in np.unique(T_labels)])

    #plotting the mean absolute value of the second component vs T
    plt.figure(figsize=(10, 6))
    plt.scatter(T_labels, mean_abs_spc, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    #line in the Tc
    plt.axvline(x=2.269, color='red', linestyle='--', label='Critical Temperature Tc')
    plt.legend()
    plt.xlabel('Temperature T', fontsize=12)
    plt.ylabel('Mean Absolute Value of Second PC', fontsize=12)
    plt.title('Mean Absolute Second PC vs Temperature', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #save in figures folder
    plt.savefig("figures/PCA_second_component_vs_T.png", dpi=300)
    plt.close()

