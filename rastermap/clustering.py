import numpy as np
from scipy.stats import zscore
import warnings
from .sorting import compute_BBt, matrix_matching, travelling_salesman


def kmeans_init(X, n_clusters=100, n_local_trials=None):
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
    n_local_trials = min(2 + 10 * int(np.log(n_clusters)), n_samples - 1)

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


def scaled_kmeans(X, n_clusters=100, n_iter=50, init="kmeans++", random_state=0):
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
        X_nodes = kmeans_init(X, n_clusters=n_clusters, n_local_trials=100)
    else:
        np.random.seed(random_state)
        X_nodes = np.random.randn(n_clusters, n_features) * (X**2).sum(axis=0)**0.5
    X_nodes = X_nodes / (1e-4 + (X_nodes**2).sum(axis=1)[:, np.newaxis])**.5

    # iterate and reassign neurons
    for j in range(n_iter):
        cc = X @ X_nodes.T
        imax = np.argmax(cc, 1)
        cc = cc * (cc > np.max(cc, 1)[:, np.newaxis] - 1e-6)
        X_nodes = cc.T @ X
        X_nodes = X_nodes / (1e-10 + (X_nodes**2).sum(axis=1)[:, np.newaxis])**.5
    X_nodes_norm = (X_nodes**2).sum(axis=1)**0.5
    X_nodes = X_nodes[X_nodes_norm > 0]
    X_nodes = X_nodes[X_nodes[:, 0].argsort()]
    cc = X @ X_nodes.T
    imax = cc.argmax(axis=1)

    if X_nodes.shape[0] < n_clusters and init == "kmeans++":
        X_nodes, imax = scaled_kmeans(X, n_clusters=n_clusters, n_iter=n_iter,
                                      init="random", random_state=random_state)

    if X_nodes.shape[0] < n_clusters and init != "kmeans++":
        warnings.warn(
            "found fewer clusters than user specified, try reducing n_clusters and/or reduce n_splits and/or increase n_PCs"
        )

    return X_nodes, imax


def kmeans(X, n_clusters=100, random_state=0):
    from sklearn.cluster import KMeans
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


def compute_cc_tdelay(U, V, U_nodes, time_lag_window=5, symmetric=False):
    """ compute correlation matrix of clusters at time offsets and take max """
    Vnorm = (V**2).sum(axis=1)**0.5
    X_nodes = U_nodes @ (V / Vnorm[:, np.newaxis])
    X_nodes = zscore(X_nodes, axis=1)
    n_nodes, nt = X_nodes.shape

    tshifts = np.arange(-time_lag_window * symmetric, time_lag_window + 1)
    cc_tdelay = np.zeros((n_nodes, n_nodes, len(tshifts)), np.float32)
    for i, tshift in enumerate(tshifts):
        if tshift < 0:
            cc_tdelay[:, :, i] = (X_nodes[:, :nt + tshift] @ X_nodes[:, -tshift:].T) / (
                nt - tshift)
        else:
            cc_tdelay[:, :,
                      i] = (X_nodes[:, tshift:] @ X_nodes[:, :nt - tshift].T) / (nt -
                                                                                 tshift)
    # sigma = time_lag_window
    # weights = np.exp(- tshifts**2 / (2*sigma**2))
    # weights /= weights.sum()
    # return (cc_tdelay * weights).sum(axis=-1)
    return cc_tdelay.max(axis=-1)


def cluster_split_and_sort(U, V=None, n_clusters=100, nc=25, n_splits=0,
                           time_lag_window=0, symmetric=False, locality=0.0,
                           scaled=True, sticky=True, U_nodes=None, verbose=True,
                           verbose_sorting=False):
    if U_nodes is None:
        if scaled:
            U_nodes, imax = scaled_kmeans(U, n_clusters=n_clusters)
        else:
            U_nodes, imax = kmeans(U, n_clusters=n_clusters)

    if time_lag_window > 0 and V is not None:
        cc = compute_cc_tdelay(U, V, U_nodes, time_lag_window=time_lag_window,
                               symmetric=symmetric)
    else:
        cc = U_nodes @ U_nodes.T

    cc, inds = travelling_salesman(cc, verbose=verbose_sorting, locality=locality,
                                   n_skip=None)[:2]
    U_nodes = U_nodes[inds]

    n_PCs = U_nodes.shape[1]
    ineurons = (U @ U_nodes.T).argmax(axis=1)
    for k in range(n_splits):
        U_nodes_new = np.zeros((0, n_PCs))
        n_nodes = U_nodes.shape[0]
        if not sticky and k > 0:
            ineurons = (U @ U_nodes.T).argmax(axis=1)
        ineurons_new = -1 * np.ones(U.shape[0], np.int64)
        for i in range(n_nodes // nc):
            ii = np.arange(n_nodes)
            node_set = np.logical_and(ii >= i * nc, ii < (i + 1) * nc)
            in_set = np.logical_and(ineurons >= i * nc, ineurons < (i + 1) * nc)
            U_nodes0, ineurons_set = kmeans(U[in_set], n_clusters=2 * nc)
            cc = U_nodes0 @ U_nodes0.T
            cc_add = U_nodes0 @ U_nodes[~node_set].T
            ifrac = node_set.mean()
            x = np.linspace(i * nc / n_nodes, (i + 1) * nc / n_nodes, 2 * nc + 1)[:-1]
            y = np.linspace(0, 1, n_nodes + 1)[:-1][~node_set]
            BBt = compute_BBt(x, x, locality=locality)
            BBt -= np.diag(np.diag(BBt))

            BBt_add = compute_BBt(x, y, locality=locality)

            cc_out, inds, seg_len = matrix_matching(cc, BBt, cc_add, BBt_add,
                                                    verbose=False)  #cc.shape[0]-25)
            U_nodes0 = U_nodes0[inds]
            ineurons_new[in_set] = 2 * nc * i + (U[in_set] @ U_nodes0.T).argmax(axis=1)
            U_nodes_new = np.vstack((U_nodes_new, U_nodes0))
        n_nodes = U_nodes_new.shape[0]
        U_nodes = U_nodes_new.copy()
        ineurons = ineurons_new.copy()
    Y_nodes = np.arange(0, U_nodes.shape[0])[:, np.newaxis]

    if not sticky:
        ineurons = (U @ U_nodes.T).argmax(axis=1)
    return U_nodes, Y_nodes, cc, ineurons
