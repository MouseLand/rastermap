import time 
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy.stats import zscore
from clustering import scaled_kmeans, kmeans
from travelling import travelling_salesman
from upsampling import grid_upsampling, quadratic_upsampling1D, upsample_grad
from metrics import embedding_quality

def embed_clusters(X_nodes, n_components=2):
    W = PCA(n_components=n_components).fit_transform(X_nodes)*0.01
    Xz = X_nodes.copy()
    Y_nodes = W/0.01

    model = TSNE(n_components=n_components, 
                init=W, 
                perplexity=100).fit(Xz) 
    Y_nodes = model.embedding_
    
    #Xz = Xz - Xz.mean(1)[:,np.newaxis]
    #Xz = Xz / (1e-3 + (Xz**2).mean(1)[:,np.newaxis])**.5
    #cc_nodes = (Xz @ Xz.T) / X_nodes.shape[1]
    #cc_nodes = np.maximum(0, 1 - cc_nodes)
    #model = TSNE(n_components=2, 
    #             init=W, 
    #             perplexity=100, 
    #             metric='precomputed').fit(cc_nodes)
    return Y_nodes

def embedding_landmarks(X, n_clusters=201, n_components=1, travelling=True):
    #X_nodes = procrustean_kmeans(X, n_clusters=n_clusters)
    #X_nodes = scaled_kmeans(X, n_clusters=n_clusters, n_iter=100)
    X_nodes = kmeans(X, n_clusters=n_clusters)
    if n_components==1 and travelling:
        igood = (X_nodes**2).sum(axis=1)> 0.9
        cc = X_nodes[igood] @ X_nodes[igood].T
        cc,inds,seg_len = travelling_salesman(cc, verbose=False)
        Y_nodes_igood = np.zeros((len(inds), 1))
        Y_nodes_igood[inds, 0] = np.arange(len(inds))
        Y_nodes = -1 * np.ones((n_clusters, 1))
        Y_nodes[igood] = Y_nodes_igood
    else:
        Y_nodes = embed_clusters(X_nodes, n_components=n_components)
    return X_nodes, Y_nodes

class Rastermap:
    """Rastermap embedding algorithm
    Rastermap takes the nPCs (400 default) of the data, and embeds them into
    n_X clusters. It returns upsampled cluster identities (n_X*upsamp).
    Optionally, a 1D embeddding is also computed across the second dimension (n_Y>0),
    smoothed over, and the PCA recomputed before fitting Rastermap.

    data : n_samples x n_features

    Parameters
    -----------
    n_components : int, optional (default: 1)
        dimension of the embedding space
    alpha : float, optional (default: 1.0)
        exponent of the power law enforced on component n as: 1/(K+n)^alpha
    n_X :  int, optional (default: 40)
        size of the grid on which the Fourier modes are rasterized
    nPC : int, optional (default: 200)
        number of PCs to use during optimization
    verbose: bool (default: True)
        whether to output progress during optimization
    """
    def __init__(self, n_components=1, n_clusters=201, grid_upsample=10,
                 nPC = 200, init='pca', alpha = 1., sorting_algorithm='travelling_salesman', 
                 quadratic_upsample=False, keep_norm_X=False, metrics=False, verbose = True):

        self.n_components = n_components
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.nPC = nPC
        self.sorting_algorithm = sorting_algorithm
        self.grid_upsample = grid_upsample
        self.quadratic_upsample = quadratic_upsample
        self.keep_norm_X = keep_norm_X
        self.metrics = metrics
        self.verbose = verbose

        self.gradient_upsample = False ### NOT IMPLEMENTED

    def fit_transform(self, X, u=None):
        """Fit X into an embedded space and return that transformed
        output.
        Inputs
        ----------
        X : array, shape (n_samples, n_features). X contains a sample per row.

        Returns
        -------
        embedding : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X, u)
        return self.embedding

    def fit(self, data=None, u=None, itrain=None):
        """Fit X into an embedded space.
        Inputs
        ----------
        X : array, shape (n_samples, n_features)
        u,s,v : svd decomposition of X (optional)

        Assigns
        ----------
        embedding : array-like, shape (n_samples, n_components)
            Stores the embedding vectors.
        u,sv,v : singular value decomposition of data S, potentially with smoothing
        isort1 : sorting along first dimension of matrix
        isort2 : sorting along second dimension of matrix (if n_Y > 0)
        cmap: correlation of each item with all locations in the embedding map (before upsampling)
        A:    PC coefficients of each Fourier mode

        """
        t0 = time.time()

        if ((u is None)):
            ### compute svd and keep iPC's of data
            
            # normalize X
            X = zscore(data, axis=1) 
            X -= X.mean(axis=0)

            nmin = np.amin(X.shape)
            nmin = np.minimum(nmin, self.nPC)
            self.nPCs = nmin
            if itrain is not None:
                Vpca = TruncatedSVD(n_components=nmin).fit_transform(X[:,itrain])
            else:
                Vpca = TruncatedSVD(n_components=nmin).fit_transform(X)
            U = Vpca / (Vpca**2).sum(axis=0)**0.5
            if itrain is not None:
                self.X_test = U @ (U.T @ X[:,~itrain])
            self.U = Vpca

            if self.keep_norm_X:
                self.X = X

            print('nPCs = {0} computed, time {1:0.2f}'.format(self.nPCs, time.time() - t0))

        else:
            self.U = u
            self.nPCs = self.U.shape[1]
            print('nPCs = {0} precomputed'.format(self.nPCs))


        X_nodes, Y_nodes = embedding_landmarks(self.U, 
                                                n_clusters=self.n_clusters, 
                                                n_components=self.n_components, 
                                                travelling=(True 
                                                            if self.sorting_algorithm=='travelling_salesman'
                                                            else False)
                                              )
        print('landmarks computed and embedded, time {0:0.2f}'.format(time.time() - t0))


        self.X_nodes = X_nodes 
        self.Y_nodes = Y_nodes
        self.n_X = int(self.n_clusters**(1/self.n_components) * max(2, self.grid_upsample))

        Y, cc, g, Xrec = grid_upsampling(self.U, X_nodes, Y_nodes, 
                                         n_X=self.n_X, 
                                         n_neighbors=min(8, self.n_clusters//4))
        print('grid upsampled, time {0:0.2f}'.format(time.time() - t0))
        
        self.embedding_grid = Y
        
        if self.quadratic_upsample:
            Y = quadratic_upsampling1D(cc, g)
        elif self.gradient_upsample:
            Y = upsample_grad(cc, self.n_components, self.n_X)
        
        isort = Y[:,0].argsort()         

        if itrain is not None and self.metrics:
            mnn, mnn_global, rho = embedding_quality(self.X_test, Y, wrapping=False)
            print(f'METRICS: local: {mnn:0.3f}; medium: {mnn_global:0.3f}; global: {rho:0.3f}')

        self.isort = isort
        self.embedding = Y
        return self