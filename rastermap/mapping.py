import time 
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np
from scipy.stats import zscore
from .clustering import kmeans, travelling_salesman, cluster_split_and_sort
from .upsampling import grid_upsampling, kriging_upsampling, quadratic_upsampling1D, upsample_grad
from .metrics import embedding_quality
from .utils import bin1d


def embedding_landmarks(X, n_clusters=50, n_components=1, travelling=True, alpha=1):
    X_nodes, imax = kmeans(X, n_clusters=n_clusters)
    if n_components==1 and travelling:
        cc = X_nodes @ X_nodes.T
        cc,inds,seg_len = travelling_salesman(cc, verbose=False, alpha=alpha)
        Y_nodes = np.arange(len(inds))[:,np.newaxis]
        X_nodes = X_nodes[inds]
    else:
        Y_nodes = embed_clusters(X_nodes, n_components=n_components)
    return X_nodes, Y_nodes, imax

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
        (any number above 150 will be very slow due to NP-hard sorting problem)
    n_splits : int, optional (default: 0)
        number of splits to perform to increase cluster size (increases local neighborhood preservation);
        (e.g. n_splits=3, n_clusters=100 => 800 clusters)
    time_lag_window : int, optional (default: 0)
        width of gaussian to weight correlations of clusters across time lags for sequence data
    grid_upsample : int, optional (default: 10)
        how much to upsample clusters
    smoothness : int, optional (default: 1)
        how much to smooth over clusters when upsampling, number from 1 to number of clusters
    n_splits : int, optional (default: 4)
        split, recluster and sort n_splits times (4 with 50 clusters => 800 clusters)
    bin_size : int, optional (default: 50)
        binning of data across n_samples to return embedding figure, X_embedding
    verbose: bool (default: True)
        whether to output progress during optimization
    """
    def __init__(self, n_clusters=100, n_splits=0, time_lag_window=0.0, ts=0.0, smoothness=1, grid_upsample=10, sticky=True,
                 n_PCs = 200, scaled_kmeans=True, bin_size=50, symmetric=False, keep_norm_X=False, verbose = True):

        self.n_components = 1 ### ONLY IN 1D
        self.n_clusters = n_clusters
        self.n_PCs = n_PCs
        self.n_splits = n_splits
        self.time_lag_window = time_lag_window
        self.symmetric = symmetric
        self.ts = ts
        self.smoothness = smoothness
        self.grid_upsample = grid_upsample
        self.bin_size = bin_size
        self.scaled_kmeans = scaled_kmeans  
        self.sticky = sticky

        self.keep_norm_X = keep_norm_X
        self.verbose = verbose

        self.sorting_algorithm = 'travelling_salesman'
        self.quadratic_upsample = False
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

    def fit(self, data=None, u=None, v=None, itrain=None, time_bin=0, normalize=True,
            compute_X_embedding=True, compute_metrics=False):
        """Fit X into an embedded space.
        Inputs
        ----------
        X : array, shape (n_samples, n_features)
        u, v : svd decomposition of X (optional), u should be u*sv

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
            if normalize:
                X = zscore(data, axis=1) 
                X -= X.mean(axis=0)
            else:
                X = data

            nmin = np.min(X.shape) - 1 
            nmin = min(nmin, self.n_PCs)
            self.n_PCs = nmin
            if itrain is not None:
                Vpca = TruncatedSVD(n_components=nmin, 
                                    random_state=0).fit_transform(bin1d(X[:,itrain], bin_size=time_bin, axis=1))
            else:
                Vpca = TruncatedSVD(n_components=nmin, 
                                    random_state=0).fit_transform(bin1d(X, bin_size=time_bin, axis=1))
            U = Vpca #/ (Vpca**2).sum(axis=0)**0.5
            if itrain is not None:
                self.X_test = U @ (U.T @ bin1d(X[:,~itrain], bin_size=time_bin, axis=1))
            self.U = Vpca

            if self.keep_norm_X:
                self.X = X
            pc_time = time.time() - t0
            print('n_PCs = {0} computed, time {1:0.2f}'.format(self.n_PCs, pc_time))

        else:
            self.U = u
            pc_time = 0
            if data is not None:
                # normalize X
                if normalize:
                    X = zscore(data, axis=1) 
                    X -= X.mean(axis=0)
                else:
                    X = data
                if itrain is not None:
                    self.X_test = self.U @ (self.U.T @ X[:,~itrain])

            self.n_PCs = self.U.shape[1]
            print('n_PCs = {0} precomputed'.format(self.n_PCs))


        self.V = None
        if self.time_lag_window > 0:
            self.V = v if v is not None else self.U.T @ X 
            
        U_nodes, Y_nodes, cc, imax = cluster_split_and_sort(self.U, 
                                                V=self.V,
                                                n_clusters=self.n_clusters, 
                                                n_splits=self.n_splits,
                                                time_lag_window=self.time_lag_window,
                                                symmetric=self.symmetric,
                                                ts=self.ts,
                                                scaled=self.scaled_kmeans,
                                                sticky=self.sticky
                                                )
        self.cc = cc
        print('landmarks computed and embedded, time {0:0.2f}'.format(time.time() - t0))

        self.embedding_clust = imax
        self.U_nodes = U_nodes 
        
        if self.time_lag_window > 0:
            Vnorm = (self.V**2).sum(axis=1)**0.5
            self.X_nodes = U_nodes @ (self.V / Vnorm[:,np.newaxis])
        elif data is not None:
            self.X_nodes = U_nodes @ (self.U.T @ X)

        self.Y_nodes = Y_nodes

        self.n_clusters = U_nodes.shape[0]
        self.n_X = int(self.n_clusters * max(2, self.grid_upsample))
        n_neighbors = max(min(8, self.n_clusters-1), self.n_clusters//5)
        e_neighbor = max(1, min(self.smoothness, n_neighbors-1))
        
        Y, corr, g, Xrec = grid_upsampling(self.U, U_nodes, Y_nodes, 
                                         n_X=self.n_X, n_neighbors=n_neighbors,
                                         e_neighbor=e_neighbor)
        print('grid upsampled, time {0:0.2f}'.format(time.time() - t0))
        
        self.U_upsampled = Xrec.copy()
        if data is not None:
            self.X_upsampled = Xrec @ (self.U.T @ X)
        self.embedding_grid = Y
        
        if self.quadratic_upsample:
            Y = quadratic_upsampling1D(cc, g)
        elif self.gradient_upsample:
            Y = upsample_grad(cc, self.n_components, self.n_X)
        
        isort = Y[:,0].argsort()         

        if itrain is not None and compute_metrics:
            mnn, mnn_global, rho = embedding_quality(self.X_test, Y, wrapping=False)
            print(f'METRICS: local: {mnn:0.3f}; medium: {mnn_global:0.3f}; global: {rho:0.3f}')

        self.isort = isort
        self.embedding = Y

        if data is not None and compute_X_embedding:
            if X.shape[0] < self.bin_size or (self.bin_size==50 and X.shape[0] < 1000):
                bin_size = X.shape[0]//20
            else:
                bin_size = self.bin_size
        
            self.X_embedding = zscore(bin1d(X[isort], bin_size), axis=1)

        self.pc_time = pc_time 
        self.map_time = time.time() - t0 - pc_time

        return self