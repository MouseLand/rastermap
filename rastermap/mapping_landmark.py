import time 
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy.stats import zscore
from .clustering import scaled_kmeans, kmeans
from .travelling import travelling_salesman
from .upsampling import grid_upsampling, quadratic_upsampling1D, upsample_grad
from .metrics import embedding_quality

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

def embedding_landmarks(X, n_clusters=201, n_components=1, travelling=True, alpha=1):
    #X_nodes = scaled_kmeans(X, n_clusters=n_clusters, n_iter=100)
    X_nodes = kmeans(X, n_clusters=n_clusters)
    if n_components==1 and travelling:
        igood = (X_nodes**2).sum(axis=1)> 0.9
        cc = X_nodes[igood] @ X_nodes[igood].T
        cc,inds,seg_len = travelling_salesman(cc, verbose=False, alpha=alpha)
        Y_nodes_igood = np.zeros((len(inds), 1))
        Y_nodes_igood[inds, 0] = np.arange(len(inds))
        Y_nodes = -1 * np.ones((n_clusters, 1))
        Y_nodes[igood] = Y_nodes_igood
    else:
        Y_nodes = embed_clusters(X_nodes, n_components=n_components)
    return X_nodes, Y_nodes

class Rastermap:
    """Rastermap embedding algorithm
    Rastermap takes the n_PCs (200 default) of the data, and embeds them into
    n_clusters clusters. It then sorts the clusters and upsamples to a grid with 
    grid_upsample * n_clusters nodes. Each data sample is assigned to a node. 
    The assignment of the samples to nodes is returned.

    data : n_samples x n_features

    Parameters
    -----------
    n_PCs : int, optional (default: 200)
        number of PCs to use during optimization
    n_clusters : int, optional (default: 100)
        number of clusters created from data before upsampling and creating embedding
    grid_upsample : int, optional (default: 10)
        how much to upsample clusters
    smoothness : int, optional (default: 1)
        how much to smooth over clusters when upsampling, number from 1 to number of clusters
    alpha : float, optional (default: 1.0)
        exponent of the power law enforced on travelling salesman algorithm for sorting clusters
    verbose: bool (default: True)
        whether to output progress during optimization
    """
    def __init__(self, n_clusters=100, smoothness=1, grid_upsample=10,
                 n_PCs = 200, alpha = 1., sorting_algorithm='travelling_salesman', 
                 quadratic_upsample=False, keep_norm_X=False, metrics=False, verbose = True):

        self.n_components = 1 ### ONLY IN 1D
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.n_PCs = n_PCs
        self.smoothness = smoothness
        self.grid_upsample = grid_upsample

        self.sorting_algorithm = sorting_algorithm
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
        isort : sorting along first dimension of matrix
        
        """
        t0 = time.time()

        if ((u is None)):
            ### compute svd and keep iPC's of data
            
            # normalize X
            X = zscore(data, axis=1) 
            X -= X.mean(axis=0)

            nmin = np.min(X.shape) - 1 
            nmin = min(nmin, self.n_PCs)
            self.n_PCs = nmin
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
            pc_time = time.time()
            print('n_PCs = {0} computed, time {1:0.2f}'.format(self.n_PCs, pc_time - t0))

        else:
            self.U = u
            pc_time = 0
            if itrain is not None:
                # normalize X
                X = zscore(data, axis=1) 
                X -= X.mean(axis=0)
                self.X_test = U @ (U.T @ X[:,~itrain])

            self.n_PCs = self.U.shape[1]
            print('n_PCs = {0} precomputed'.format(self.n_PCs))


        U_nodes, Y_nodes = embedding_landmarks(self.U, 
                                                n_clusters=self.n_clusters, 
                                                n_components=self.n_components, 
                                                travelling=(True 
                                                            if self.sorting_algorithm=='travelling_salesman'
                                                            else False),
                                                alpha=self.alpha
                                              )
        print('landmarks computed and embedded, time {0:0.2f}'.format(time.time() - t0))


        self.U_nodes = U_nodes 
        self.X_nodes = U_nodes @ (U.T @ X)
        self.Y_nodes = Y_nodes
        self.n_X = int(self.n_clusters**(1/self.n_components) * max(2, self.grid_upsample))

        n_neighbors = max(min(8, self.n_clusters-1), self.n_clusters//5)
        e_neighbor = max(1, min(self.smoothness, n_neighbors-1))
        Y, cc, g, Xrec = grid_upsampling(self.U, U_nodes, Y_nodes, 
                                         n_X=self.n_X, 
                                         n_neighbors=n_neighbors,
                                         e_neighbor=e_neighbor)
        print('grid upsampled, time {0:0.2f}'.format(time.time() - t0))
        
        self.U_upsampled = Xrec.copy()
        self.X_upsampled = Xrec @ (U.T @ X)
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

        self.pc_time = pc_time 
        self.map_time = time.time() -t0 - pc_time

        return self