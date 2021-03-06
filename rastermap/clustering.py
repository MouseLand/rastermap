import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import zscore

def scaled_kmeans(X, n_clusters=201, n_iter = 20):
    n_samples, n_features = X.shape
    X_nodes = (np.random.randn(n_clusters, n_features) / 
                                (1 + np.arange(n_features))**0.5)
    X_nodes = X_nodes / (1e-4 + np.sum(X_nodes**2, 1)[:,np.newaxis])**.5
    for j in range(n_iter):
        cc = X @ X_nodes.T 
        imax = np.argmax(cc, 1)
        cc = cc * (cc > np.max(cc, 1)[:,np.newaxis]-1e-6)
        X_nodes = cc.T @ X 
        X_nodes = X_nodes / (1e-10 + np.sum(X_nodes**2, 1)[:,np.newaxis])**.5  
    return X_nodes

def embed_clusters(X_nodes):
    W = PCA(n_components=2).fit_transform(X_nodes)*0.01
    Xz = X_nodes.copy()
    model = TSNE(n_components=2, 
                init=W, 
                perplexity=100).fit(Xz) 

    #Xz = Xz - Xz.mean(1)[:,np.newaxis]
    #Xz = Xz / (1e-3 + (Xz**2).mean(1)[:,np.newaxis])**.5
    #cc_nodes = (Xz @ Xz.T) / X_nodes.shape[1]
    #cc_nodes = np.maximum(0, 1 - cc_nodes)
    #model = TSNE(n_components=2, 
    #             init=W, 
    #             perplexity=100, 
    #             metric='precomputed').fit(cc_nodes)
    #Y_nodes = model.embedding_
    Y_nodes = W/0.01
    return Y_nodes

def embedding_landmarks(X, n_clusters=201):
    X_nodes = scaled_kmeans(X, n_clusters=n_clusters)
    Y_nodes = embed_clusters(X_nodes)
    return X_nodes, Y_nodes

