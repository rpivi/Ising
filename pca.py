from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import numpy as np


def perform_pca(spins_configs, n_components=2):
    #spins configs is a list of 2D arrays of shape (L, L) for different BJ values
    # Flatten each 2D configuration into a 1D vector
    X = []
    for config in spins_configs:
        X.append(config.flatten())
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_

def pca_plot(X_pca, BJ_s, spins_configs):

    BJ_labels = [] 
    for beta, configs in zip(BJ_s, spins_configs):
        for cfg in configs:
            BJ_labels.append(beta)

    BJ_labels = np.array(BJ_labels)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=BJ_labels, cmap='viridis')
    plt.colorbar(scatter, label='BJ')
    plt.title("PCA of Spin Configurations")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.savefig("PCA_spin_configurations.png")
    plt.close()