import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore


def kmeans(X, n_clusters=100):
    X_norm = 1 #(1e-10 + (X**2).sum(axis=0)**.5)
    model = KMeans(n_init=1, n_clusters=n_clusters, random_state=0).fit(X / X_norm)
    X_nodes = model.cluster_centers_ * X_norm
    X_nodes = X_nodes / (1e-10 + ((X_nodes**2).sum(axis=1))[:,np.newaxis])**.5  
    cc = X @ X_nodes.T 
    imax = cc.argmax(axis=1)
    return X_nodes, imax

def scaled_kmeans(X, n_clusters=201, n_iter = 20):
    n_samples, n_features = X.shape
    X_nodes = np.random.randn(n_clusters, n_features) #/ 
    #                            (1 + np.arange(n_features))**0.5)
    X_nodes = X_nodes / (1e-4 + np.sum(X_nodes**2, 1)[:,np.newaxis])**.5
    for j in range(n_iter):
        cc = X @ X_nodes.T 
        imax = cc.argmax(axis=1)
        cc = cc * (cc > np.max(cc, 1)[:,np.newaxis]-1e-6)
        X_nodes = cc.T @ X 
        X_nodes = X_nodes / (1e-10 + np.sum(X_nodes**2, 1)[:,np.newaxis])**.5  
    return X_nodes, imax
