from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np

def prepare_pca_data(spins_configs_nested, BJ_s):
    all_configs = []
    BJ_labels = []
    
    for BJ, configs in zip(BJ_s, spins_configs_nested):
        all_configs.extend(configs)
        BJ_labels.extend([BJ] * len(configs))
    
    return all_configs, BJ_labels


def perform_pca(configs_flat, n_components=2):
    # Appiattisci ogni configurazione 2D in vettore 1D
    X = np.array([config.flatten() for config in configs_flat])
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    return X_pca, pca.explained_variance_ratio_

def pca_plot(X_pca, BJ_labels, explained_variance_ratio=None):
    BJ_labels = np.array(BJ_labels)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=BJ_labels, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, label='β/J')
    
    if explained_variance_ratio is not None:
        xlabel = f"PC1 ({explained_variance_ratio[0]*100:.1f}% var)"
        ylabel = f"PC2 ({explained_variance_ratio[1]*100:.1f}% var)"
    else:
        xlabel = "Principal Component 1"
        ylabel = "Principal Component 2"
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title("PCA of Spin Configurations - Ising Model", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("PCA_spin_configurations.png", dpi=300)
    plt.close()