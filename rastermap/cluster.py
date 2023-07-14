"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import warnings
from sklearn.cluster import MiniBatchKMeans as KMeans
from scipy.stats import zscore

def _scaled_kmeans_init(X, n_clusters=100, n_local_trials=None):
    """Init n_clusters seeds according to k-means++ using correlations
    
    adapted from scikit-learn's code for correlation distance
    
    Parameters
    -----------
    X : array shape (n_samples, n_features)
        The data to pick seeds for. 
    n_clusters : integer
        The number of seeds to choose
    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -----------
    X_nodes : array shape (n_clusters, n_features)
        cluster centers for initializing kmeans
    
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape
    X_nodes = np.empty((n_clusters, n_features), dtype=X.dtype)
    if n_local_trials is None:
        n_local_trials = min(2 + 10 * int(np.log(n_clusters)), n_samples - 1)
    else:
        n_local_trials = min(n_samples-1, n_local_trials)

    # Pick first center randomly
    center_id = np.random.randint(n_samples)
    X_nodes[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    X_norm = (X**2).sum(axis=1)**0.5
    closest_dist_sq = 1 - (X @ X_nodes[0]) / (X_norm * (X_nodes[0]**2).sum()**0.5)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq.astype(np.float64)),
                                        rand_vals)

        # Compute distances to center candidates
        X_candidates = X[candidate_ids]
        X_candidates_norm = (X_candidates**2).sum(axis=1)**0.5
        distance_to_candidates = 1 - (X @ X_candidates.T) / np.outer(
            X_norm, X_candidates_norm)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidates[:, trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        X_nodes[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return X_nodes


def scaled_kmeans(X, n_clusters=100, n_iter=50, n_local_trials=100, 
                    init="kmeans++", random_state=0):
    """ kmeans using correlation distance
    
    Parameters
    -----------
    X : array shape (n_samples, n_features)
        The data to cluster
    n_clusters : integer, optional (default=100)
        The number of clusters
    n_iter : integer, optional (default=50)
        number of iterations
    random_state : integer, optional (default=0)
        seed of numpy random number generator

    Returns
    -----------
    X_nodes : array shape (n_clusters, n_features)
        cluster centers found by kmeans
    imax : array (n_samples)
        best cluster for each data point

    """
    n_samples, n_features = X.shape
    # initialize with kmeans++
    if init == "kmeans++":
        np.random.seed(random_state)
        X_nodes = _scaled_kmeans_init(X, n_clusters=n_clusters, 
                                        n_local_trials=n_local_trials)
    else:
        np.random.seed(random_state)
        X_nodes = np.random.randn(n_clusters, n_features) * (X**2).sum(axis=0)**0.5
    X_nodes = X_nodes / (1e-4 + (X_nodes**2).sum(axis=1)[:, np.newaxis])**.5
    
    # iterate and reassign neurons
    for j in range(n_iter):
        cc = X @ X_nodes.T
        imax = np.argmax(cc, axis=1)
        cc = cc * (cc > np.max(cc, 1)[:, np.newaxis] - 1e-6)
        X_nodes = cc.T @ X
        X_nodes = X_nodes / (1e-10 + (X_nodes**2).sum(axis=1)[:, np.newaxis])**.5
    X_nodes_norm = (X_nodes**2).sum(axis=1)**0.5
    X_nodes = X_nodes[X_nodes_norm > 0]
    X_nodes = X_nodes[X_nodes[:, 0].argsort()]
    
    if X_nodes.shape[0] < n_clusters // 2 and init == "kmeans++":
        warnings.warn(
            "found fewer than half the n_clusters that the user specified, rerunning with random initialization"
        )
        X_nodes, imax = scaled_kmeans(X, n_clusters=n_clusters, n_iter=n_iter,
                                      init="random", random_state=random_state)
    else:
        cc = X @ X_nodes.T
        imax = cc.argmax(axis=1)

    if X_nodes.shape[0] < n_clusters and init != "kmeans++":
        warnings.warn(
            "found fewer clusters than user specified, try reducing n_clusters and/or reduce n_splits and/or increase n_PCs"
        )

    return X_nodes, imax

def kmeans(X, n_clusters=100, random_state=0):
    np.random.seed(random_state)
    #X_nodes = (np.random.randn(n_clusters, n_features) /
    #                            (1 + np.arange(n_features))**0.5)
    #X_nodes = X_nodes / (1e-4 + (X_nodes**2).sum(axis=1)[:,np.newaxis])**.5
    model = KMeans(n_init=1, init="k-means++", n_clusters=n_clusters,
                   random_state=0).fit(X)
    X_nodes = model.cluster_centers_
    X_nodes = X_nodes / (1e-10 + ((X_nodes**2).sum(axis=1))[:, np.newaxis])**.5
    imax = model.labels_
    X_nodes = X_nodes[X_nodes[:, 0].argsort()]
    cc = X @ X_nodes.T
    imax = cc.argmax(axis=1)
    return X_nodes, imax

def compute_cc_tdelay(V, U_nodes, time_lag_window=5, symmetric=False):
    """ compute correlation matrix of clusters at time offsets and take max """
    X_nodes = U_nodes @ V.T
    X_nodes = zscore(X_nodes, axis=1)
    n_nodes, nt = X_nodes.shape

    tshifts = np.arange(-time_lag_window * symmetric, time_lag_window + 1)
    cc_tdelay = np.zeros((n_nodes, n_nodes, len(tshifts)), np.float32)
    for i, tshift in enumerate(tshifts):
        if tshift < 0:
            cc_tdelay[:, :, i] = ((X_nodes[:, :nt + tshift] @ X_nodes[:, -tshift:].T) / 
                                    (nt - tshift))
        else:
            cc_tdelay[:, :, i] = ((X_nodes[:, tshift:] @ X_nodes[:, :nt - tshift].T) / 
                                    (nt -  tshift))

    return cc_tdelay.max(axis=-1)