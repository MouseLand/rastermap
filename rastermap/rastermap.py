"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import time
import numpy as np
import warnings
from scipy.stats import zscore

from .cluster import kmeans, scaled_kmeans, compute_cc_tdelay
from .sort import traveling_salesman, compute_BBt, matrix_matching
from .upsample import grid_upsampling
from .utils import bin1d
from .svd import SVD

def default_settings():
    """ default settings for main algorithm settings """
    settings = {}
    settings["n_clusters"] = 100
    settings["n_PCs"] = 200
    settings["time_lag_window"] = 0
    settings["locality"] = 0.0
    settings["n_splits"] = 0
    settings["time_bin"] = 0
    settings["grid_upsample"] = 10
    settings["mean_time"] = True
    settings["verbose"] = True
    settings["verbose_sorting"] = False
    return settings


def sequence_settings():
    """ good for data with sequences """
    settings = default_settings()
    settings["time_lag_window"] = 10
    settings["locality"] = 0.75
    return settings


def highd_settings():
    """ good for large scale data with high-d features like natural image responses """
    settings = default_settings()
    settings["n_clusters"] = 100
    settings["n_splits"] = 3
    settings["nPCs"] = 400
    return settings


def settings_info():
    info = {}
    info["n_clusters"] = (
        """number of clusters created from data before upsampling and creating embedding \n 
            (any number above 150 will be very slow due to NP-hard sorting problem)"""
    )
    info["n_PCs"] = "number of PCs to use during optimization"
    info["time_lag_window"] = (
        """number of time points into the future to compute cross-correlation, 
            useful for sequence finding"""
    )
    info["locality"] = (
        "how local should the algorithm be -- set to 1.0 for highly local + sequence finding"
    )
    info["grid_upsample"] = "how much to upsample clusters"
    info["n_splits"] = (
        "split, recluster and sort n_splits times (increases local neighborhood preservation); \n (4 with 50 clusters => 800 clusters)"
    )
    info["time_bin"] = (
        """ binning of data in time before PCA is computed """
    )
    info["mean_time"] = (
        """whether to project out the mean over data samples at each timepoint,
             usually good to keep on to find structure"""
    )
    info["verbose"] = "whether to output progress during optimization"
    info["verbose_sorting"] = "output progress in travelling salesman"
    return info


class Rastermap:
    """Rastermap embedding algorithm
    Rastermap takes the n_PCs (200 default) of the data, and embeds them into
    n_clusters clusters. It then sorts the clusters and upsamples to a grid with 
    grid_upsample * n_clusters nodes. Each data sample is assigned to a node. 
    The assignment of the samples to nodes is attribute `embedding`, and the sorting 
    of the nodes is `isort`.

    Parameters in order of importance

    Parameters
    -----------
    n_clusters : int, optional (default: 100)
        number of clusters created from data before upsampling and creating embedding
        (any number above 150 will be slow due to NP-hard sorting problem, max is 200)
    n_PCs : int, optional (default: 200)
        number of PCs to use during optimization
    time_lag_window : int, optional (default: 0)
        number of time points into the future to compute cross-correlation, 
        useful to set to several timepoints for sequence finding
    locality : float, optional (default: 0.0)
        how local should the algorithm be -- set to 1.0 for highly local + 
        sequence finding, and 0.0 for global sorting
    grid_upsample : int, optional (default: 10)
        how much to upsample clusters, if set to 0.0 then no upsampling
    time_bin : int, optional (default: 0)
        binning of data in time before PCA is computed, if set to 0 or 1 no binning occurs
    mean_time : bool, optional (default: True)
        whether to project out the mean over data samples at each timepoint,
             usually good to keep on to find structure
    n_splits : int, optional (default: 0)
        split, recluster and sort n_splits times 
        (increases local neighborhood preservation for high-dim data); 
        results in (n_clusters * 2**n_splits) clusters
    run_scaled_kmeans : bool, optional (default: True)
        run scaled_kmeans as clustering algorithm; if False, run kmeans
    verbose : bool (default: True)
        whether to output progress during optimization
    verbose_sorting : bool (default: False)
        output progress in travelling salesman    
    keep_norm_X : bool, optional (default: True)
        keep normalized version of X saved as member of class
    bin_size : int, optional (default: 0)
        binning of data across n_samples to return embedding figure, X_embedding; 
        if 0, then binning based on data size, if 1 then no binning
    symmetric : bool, optional (default: False)
        if False, use only positive time lag cross-correlations for sorting 
        (only makes a difference if time_lag_window > 0); 
        recommended to keep False for sequence finding
    sticky : bool, optional (default: True)
        if n_splits>0, sticky=True keeps neurons in same place as initial sorting before splitting; 
        otherwise neurons can move each split (which generally does not work as well)
    nc_splits : int, optional (default: None)
        if n_splits > 0, size to split n_clusters into; 
        if None, nc_splits = min(50, n_clusters // 4)
    smoothness : int, optional (default: 1)
        how much to smooth over clusters when upsampling, number from 1 to number of 
        clusters (recommended to not change, instead use locality to change sorting)
    
    """

    def __init__(self, n_clusters=100, n_PCs=200, time_lag_window=0.0, locality=0.0,
                 grid_upsample=10, time_bin=0, normalize=True, mean_time=True, n_splits=0,
                 run_scaled_kmeans=True, verbose=True, verbose_sorting=False, 
                 keep_norm_X=True, bin_size=0, symmetric=False, 
                 sticky=True, nc_splits=None, smoothness=1):

        self.n_clusters = n_clusters
        self.n_PCs = n_PCs
        self.time_lag_window = time_lag_window
        self.locality = locality
        self.grid_upsample = grid_upsample
        self.time_bin = time_bin
        self.normalize = normalize
        self.mean_time = mean_time
        self.n_splits = n_splits
        self.run_scaled_kmeans = run_scaled_kmeans
        self.verbose = verbose
        self.verbose_sorting = verbose_sorting

        self.keep_norm_X = keep_norm_X
        self.bin_size = bin_size                
        self.symmetric = symmetric
        self.sticky = sticky
        self.nc_splits = nc_splits
        self.smoothness = smoothness
        
        self.sorting_algorithm = "travelling_salesman"
        
    def fit(self, data=None, Usv=None, Vsv=None, U_nodes=None, itrain=None,
            compute_X_embedding=False):
        """ sorts data or singular value decomposition of data by first axis

        can use full data matrix, or use singular value decomposition, to reduce RAM 
        requirements, it is recommended to use float32. see Rastermap class for 
        parameter information (`Rastermap?`)

        example usage:
        ```
        import numpy as np
        import matplotlib.pyplot as plt
        from rastermap import Rastermap, utils
        from scipy.stats import zscore

        # spks is neurons by time
        spks = np.load("spks.npy").astype("float32")
        spks = zscore(spks, axis=1)

        # fit rastermap
        model = Rastermap(n_PCs=200, n_clusters=100, 
                        locality=0.75, time_lag_window=5).fit(spks)
        y = model.embedding # neurons x 1
        isort = model.isort

        # bin over neurons
        X_embedding = zscore(utils.bin1d(spks, bin_size=25, axis=0), axis=1)

        # plot
        fig = plt.figure(figsize=(12,5))
        ax = fig.add_subplot(111)
        ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
        ```

        Inputs
        ----------
        data : array, shape (n_samples, n_features) (optional, default None) 
            this matrix is usually neurons/voxels by time, or None if using decomposition, 
            e.g. as in widefield imaging
        Usv : array, shape (n_samples, n_PCs) (optional, default None)
            singular vectors U times singular values sv
        Vsv : array, shape (n_features, n_PCs) (optional, default None)
            singular vectors U times singular values sv
        U_nodes : array, shape (n_clusters, n_PCs) (optional, default None)
            cluster centers in PC space, if you have precomputed them
        itrain : array, shape (n_features,) (optional, default None)
            fit embedding on timepoints itrain only

        Assigns
        ----------
        embedding : array, shape (n_samples, 1)
            embedding of each neuron / voxel
        isort : sorting along first dimension of input matrix
            use this to get neuron / voxel sorting
        igood : array, shape (n_samples, 1)
            neurons/voxels which had non-zero activity and were used for sorting
        Usv : array, shape (n_samples, n_PCs) 
            singular vectors U times singular values sv
        Vsv : array, shape (n_features, n_PCs)
            singular vectors U times singular values sv
        U_nodes : array, shape (n_clusters, n_PCs) 
            cluster centers in PC space
        Y_nodes : array, shape (n_clusters, 1) 
            np.arange(0, n_clusters)
        X_nodes : array, shape (n_clusters, n_features)
            cluster activity traces in time
        cc : array, shape (n_clusters, n_clusters)
            sorted asymmetric similarity matrix
        embedding_clust : array, shape (n_samples, 1)
            assignment of each neuron/voxel to each cluster (before upsampling)
        X : array, shape (n_samples, n_features)
            normalized data stored (if keep_norm_X is True)
        X_embedding : array, shape (n_samples//bin_size, n_features)
            normalized data binned across samples (if compute_X_embedding is True)

        """
        t0 = time.time()
        self.n_clusters = None if self.n_clusters==0 else self.n_clusters
        
        # normalize data
        igood = ~np.isnan(data[:,0]) if data is not None else ~np.isnan(Usv[:,0])
        stdx = None
        normed = False
        if data is not None:
            if self.normalize:
                if hasattr(self, "X"):
                    warnings.warn("not renormalizing / subtracting mean, using previous normalization")
                    X = self.X 
                else:
                    print("normalizing data across axis=1")
                    X = data.copy() 
                    if self.time_bin > 1:
                        print(f"binning in time with time_bin = {self.time_bin}")
                        X = bin1d(X, bin_size=self.time_bin, axis=1)
                    X -= X.mean(axis=1)[:,np.newaxis]
                    stdx = X.std(axis=1)
                    X /= stdx[:,np.newaxis]
                    normed = True
            else:
                if self.time_bin > 1:
                    X = bin1d(data.copy(), bin_size=self.time_bin, axis=1)
                else:
                    X = data
            if self.mean_time:
                if hasattr(self, "X"):
                    warnings.warn("not renormalizing / subtracting mean, using previous normalization")
                    X = self.X 
                else:
                    print("projecting out mean along axis=0")
                    X_mean = np.nanmean(X, axis=0, keepdims=True).T
                    X_mean /= (X_mean**2).sum()**0.5
                    w_mean = X @ X_mean
                    X -= w_mean @ X_mean.T
                    normed = True
            if self.keep_norm_X and (self.mean_time or self.normalize):
                self.X = X
        elif hasattr(self, "X"):
            X = self.X
            Vsv_sub = Vsv.copy()
            if self.normalize or self.mean_time:
                warnings.warn("not renormalizing / subtracting mean, using previous normalization")
        else:
            if self.mean_time:
                V_mean = Vsv.mean(axis=1, keepdims=True)
                V_mean /= (V_mean**2).sum()**0.5
                self.V_mean = V_mean
                proj_mean = V_mean @ (V_mean.T @ Vsv)
                Vsv_sub = Vsv.copy() - proj_mean
                normed = True
            else:
                Vsv_sub = Vsv.copy()
            stdx = Usv.std(axis=1) if stdx is None else stdx
        if normed:
            print(f"data normalized, {time.time()-t0:0.2f}sec")    

        stdx = X.std(axis=1) if stdx is None else stdx
        igood = np.logical_and(igood, stdx > 0)
        n_samples = igood.sum() 
        n_time = data.shape[1] if data is not None else Vsv.shape[0]
        print(f"sorting activity: {n_samples} valid samples by {n_time} timepoints")
        

        ### ------------- PCA ------------------------------------------------------ ###        
        if not hasattr(self, "Usv"):
            if Usv is None:
                tic = time.time()
                Usv_valid = SVD(X[igood][:, itrain] if itrain is not None else X, 
                               n_components=self.n_PCs)            
                Usv = np.nan * np.zeros((len(igood), Usv_valid.shape[1]), "float32")
                Usv[igood] = Usv_valid
                self.Usv = Usv
                self.n_PCs = Usv.shape[1]
                pc_time = time.time() - tic
                print(f"n_PCs = {self.n_PCs} computed, {time.time()-t0:0.2f}sec")    
            elif Usv is not None:
                self.Usv = Usv
                pc_time = 0
                self.n_PCs = self.Usv.shape[1]
        self.sv = np.nansum((self.Usv**2), axis=0)**0.5
        if not hasattr(self, "Vsv"):
            if Vsv is None:
                U = self.Usv.copy() / self.sv
                self.Vsv = X.T @ U 
            elif Vsv is not None:
                self.Vsv = Vsv_sub

        ### ------------- clustering ----------------------------------------------- ###
        if self.run_scaled_kmeans:
            kmeans_func = scaled_kmeans
        else:
            kmeans_func = kmeans

        if ((U_nodes is None and self.Usv.shape[0] <= 50) or 
            (self.Usv.shape[0] <= 200 and self.n_clusters is None)):
            # number of neurons / voxels <= 50, skip clustering
            if U_nodes is None and self.Usv.shape[0] <= 50:
                warnings.warn("""data has <= 50 samples, \n
                                going to skip clustering and sort samples""")
            elif (self.Usv.shape[0] <= 200 and self.n_clusters is None):
                print("skipping clustering, n_clusters is None")
            U_nodes = self.Usv[igood].copy()
            imax = np.arange(0, U_nodes.shape[0])
        elif self.n_clusters is None:
            raise ValueError("n_clusters set to None")
        elif self.n_clusters >= 200:
            raise ValueError("n_clusters cannot be greater than 200")
        elif not hasattr(self, "U_nodes") and U_nodes is not None:
            # use user input for clusters
            print("using cluster input from user")
            cu = self.Usv[igood] @ U_nodes.T
            imax = cu.argmax(axis=1)
        elif hasattr(self, "U_nodes") and self.U_nodes is not None:
            # skip clustering as it was run before
            if self.n_splits==1:
                U_nodes = self.U_nodes
                imax = self.embedding_clust
                warnings.warn("""clusters already computed, skipping clustering""")
            else:
                raise ValueError("""cannot rerun if n_splits > 0\n 
                                    need to set model.U_nodes = None first or \n 
                                    reset model = Rastermap(...) """)
        else:
            # run clustering
            self.n_clusters = min(self.Usv.shape[0]//2, self.n_clusters)
            U_nodes, imax = kmeans_func(self.Usv[igood], n_clusters=self.n_clusters)
            print(f"{U_nodes.shape[0]} clusters computed, time {time.time() - t0:0.2f}sec")

        # compute correlation matrix across clusters
        if self.time_lag_window > 0:
            cc = compute_cc_tdelay(self.Vsv / self.sv, U_nodes, 
                                   time_lag_window=self.time_lag_window,
                                   symmetric=self.symmetric)
        else:
            cc = U_nodes @ U_nodes.T

        ### ---------------- sorting ----------------------------------------------- ###
        cc, inds = traveling_salesman(cc, verbose=self.verbose_sorting, 
                                       locality=self.locality,
                                        n_skip=None)[:2]
        U_nodes = U_nodes[inds]
        ineurons = (self.Usv[igood] @ U_nodes.T).argmax(axis=1)
        self.cc = cc
        Y_nodes = np.arange(0, U_nodes.shape[0])[:, np.newaxis]

        # split and recluster and sort
        if self.n_splits > 0:
            n_nodes = U_nodes.shape[0]
            nc = min(50, n_nodes // 4) if self.nc_splits is None else self.nc_splits
            for k in range(self.n_splits):
                U_nodes_new = np.zeros((0, self.n_PCs))
                n_nodes = U_nodes.shape[0]
                if not self.sticky and k > 0:
                    ineurons = (self.Usv[igood] @ U_nodes.T).argmax(axis=1)
                ineurons_new = -1 * np.ones(self.Usv[igood].shape[0], np.int64)
                for i in range(n_nodes // nc):
                    ii = np.arange(n_nodes)
                    node_set = np.logical_and(ii >= i * nc, ii < (i + 1) * nc)
                    in_set = np.logical_and(ineurons >= i * nc, ineurons < (i + 1) * nc)
                    U_nodes0, ineurons_set = kmeans_func(self.Usv[igood][in_set], 
                                                         n_clusters=2 * nc)                                
                    cc = U_nodes0 @ U_nodes0.T
                    cc_add = U_nodes0 @ U_nodes[~node_set].T
                    ifrac = node_set.mean()
                    x = np.linspace(i * nc / n_nodes, 
                                    (i + 1) * nc / n_nodes, 
                                    cc.shape[0] + 1)[:-1]
                    y = np.linspace(0, 1, n_nodes + 1)[:-1][~node_set]
                    BBt = compute_BBt(x, x, locality=self.locality)
                    BBt -= np.diag(np.diag(BBt))
                    BBt_add = compute_BBt(x, y, locality=self.locality)
                    cc_out, inds, seg_len = matrix_matching(cc, BBt, cc_add, BBt_add,
                                                            verbose=False)  
                    U_nodes0 = U_nodes0[inds]
                    ineurons_new[in_set] = (2 * nc * i + # offset based on partition
                                            (self.Usv[igood][in_set] @ U_nodes0.T).argmax(axis=1))
                    U_nodes_new = np.vstack((U_nodes_new, U_nodes0))
                n_nodes = U_nodes_new.shape[0]
                U_nodes = U_nodes_new.copy()
                ineurons = ineurons_new.copy()
            if not self.sticky:
                ineurons = (self.Usv[igood] @ U_nodes.T).argmax(axis=1)
            Y_nodes = np.arange(0, U_nodes.shape[0])[:, np.newaxis]

        print(f"clusters sorted, time {time.time() - t0:0.2f}sec")
        
        ### ---------------- upsample ---------------------------------------------- ###
        self.n_clusters = U_nodes.shape[0]
        if self.grid_upsample > 0:
            self.n_X = int(self.n_clusters * max(2, self.grid_upsample))
            n_neighbors = max(min(8, self.n_clusters - 1), self.n_clusters // 5)
            e_neighbor = max(1, min(self.smoothness, n_neighbors - 1))

            Y, corr, g, Xrec = grid_upsampling(self.Usv[igood], U_nodes, Y_nodes, n_X=self.n_X,
                                            n_neighbors=n_neighbors,
                                            e_neighbor=e_neighbor)
            print(f"clusters upsampled, time {time.time() - t0:0.2f}sec")
        else:
            if len(U_nodes) == n_samples:
                Y = np.zeros(n_samples, "int")
                Y[inds] = np.arange(0, n_samples)
            else:
                Y = np.zeros(n_samples, "int")
                Y[ineurons.argsort()] = np.arange(0, n_samples)
            Y = Y[:,np.newaxis]
            Xrec = U_nodes
        
        self.embedding_clust = ineurons
        self.U_nodes = U_nodes
        self.Y_nodes = Y_nodes
        self.U_upsampled = Xrec.copy()
        self.embedding_valid = Y
        self.isort_valid = Y[:, 0].argsort()

        # convert cluster centers to time traces (not used in algorithm)
        Vnorm = (self.Vsv**2).sum(axis=1)**0.5
        self.X_nodes = U_nodes @ (self.Vsv / Vnorm[:, np.newaxis]).T
        # upsampled cluster centers in time (these are large so we won't keep them for now)
        # if data is not None:
        #     self.X_upsampled = Xrec @ (self.Vsv / Vnorm[:, np.newaxis]).T
        
        self.igood = igood
        self.embedding = np.nan * np.zeros((len(self.igood),1))
        self.embedding[igood] = self.embedding_valid
        self.isort = self.embedding[:,0].argsort()
        
        ### ----------- bin across embedding --------------------------------------- ###
        if data is not None and compute_X_embedding:
            if (bin_size==0 or n_samples < bin_size or 
                (bin_size == 50 and n_samples < 1000)):
                bin_size = max(1, n_samples // 500)
            self.X_embedding = zscore(bin1d(X[igood][self.isort], bin_size, axis=0), axis=1)

        print(f"rastermap complete, time {time.time() - t0:0.2f}sec")

        self.runtime = time.time() - t0
        self.pc_time = pc_time

        return self
