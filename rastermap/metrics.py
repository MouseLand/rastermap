import time
from scipy.stats import zscore, spearmanr
import multiprocessing
from multiprocessing import Pool
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import numpy as np
import scipy

def distance_matrix(Z, n_X = None, wrapping=False, correlation = False):
    if wrapping:
        #n_X = int(np.floor(Z.max() + 1))
        dists = (Z - Z[:,np.newaxis,:]) % n_X
        Zdist = (np.minimum(dists, n_X - dists)**2).sum(axis=-1)
    else:
        if correlation:
            Zdist = Z @ Z.T
            Z2 = 1e-10 + np.diag(Zdist)**.5
            Zdist = 1 - Zdist / np.outer(Z2, Z2)
        else:
            #Zdist = ((Z - Z[:,np.newaxis,:])**2).sum(axis=-1)
            Z2 = np.sum(Z**2, 1)
            Zdist = Z2 + Z2[:, np.newaxis] - 2 * Z @ Z.T
        Zdist = np.maximum(0, Zdist)

        #import pdb; pdb.set_trace();
    return Zdist

def embedding_score(X, Z, knn=[10, 100, 1000], subsetsize=5000,
                        wrapping=False, n_X = 0):

    np.random.seed(101)
    subset = np.random.choice(X.shape[0], size=subsetsize, replace=False)
    Xdist = distance_matrix(X[subset], correlation=True)
    if Z.shape[1]>2:
        Zdist = distance_matrix(Z[subset], correlation=True)
    else:
        Zdist = distance_matrix(Z[subset], n_X = n_X, wrapping=wrapping)

    mnn = []
    for kni in knn:
        nbrs1 = NearestNeighbors(n_neighbors=kni, metric='precomputed').fit(Xdist)
        ind1 = nbrs1.kneighbors(return_distance=False)

        nbrs2 = NearestNeighbors(n_neighbors=kni, metric='precomputed').fit(Zdist)
        ind2 = nbrs2.kneighbors(return_distance=False)

        intersections = 0.0
        for i in range(subsetsize):
            intersections += len(set(ind1[i]) & set(ind2[i]))
        mnn.append( intersections / subsetsize / kni )

    xd = Xdist[np.tril_indices(Xdist.shape[0],-1)]
    zd = Zdist[np.tril_indices(Xdist.shape[0],-1)]
    rho = spearmanr(xd, zd).correlation

    return mnn, rho

def embedding_quality(X, Z, classes=None, knn=20, knn_global=500, subsetsize=2000,
                        wrapping=False, n_X = 0):

    np.random.seed(101)
    subset = np.random.choice(X.shape[0], size=subsetsize, replace=False)
    Xdist = distance_matrix(X[subset], correlation=True)
    Zdist = distance_matrix(Z[subset], n_X = n_X, wrapping=wrapping)
    xd = Xdist[np.tril_indices(Xdist.shape[0],-1)]
    zd = Zdist[np.tril_indices(Xdist.shape[0],-1)]

    mnn = []
    for kni in [knn, knn_global]:
        nbrs1 = NearestNeighbors(n_neighbors=kni, metric='precomputed').fit(Xdist)
        ind1 = nbrs1.kneighbors(return_distance=False)

        nbrs2 = NearestNeighbors(n_neighbors=kni, metric='precomputed').fit(Zdist)
        ind2 = nbrs2.kneighbors(return_distance=False)

        intersections = 0.0
        for i in range(subsetsize):
            intersections += len(set(ind1[i]) & set(ind2[i]))
        mnn.append( intersections / subsetsize / kni )

    rho = spearmanr(xd, zd).correlation

    return (mnn[0], mnn[1], rho)

def split_traintest(NT, Tblock, nfrac):
    '''NT: number of timepoints, Tblock: size of blocks, nfrac: fraction of train'''
    indx = np.ceil(np.arange(0,NT) / Tblock).astype(int)
    Nblocks = indx.max()
    irand = np.random.permutation(Nblocks)
    Ntrain = int(np.ceil(nfrac * float(Nblocks)))
    Ntest = Nblocks - Ntrain
    btrain = np.sort(irand[np.arange(0,Ntrain).astype(int)])
    btest  = np.sort(irand[np.arange(Ntrain,Nblocks).astype(int)])
    itrain = np.in1d(indx, btrain)
    itest  = np.in1d(indx, btest)
    return itrain, itest

def embedding_distance(x, y, alg):
    dists = np.zeros((x.shape[0], y.shape[0]), np.float32)
    for i in range(x.shape[1]):
        dists += (x[:,i][:,np.newaxis] - y[:,i][np.newaxis,:])**2
    return dists

def corr(x,y):
    ''' correlates x[i,:] with y[i,:] for all i '''
    x -= x.mean(axis=1)[:,np.newaxis]
    y -= y.mean(axis=1)[:,np.newaxis]
    cc = (x * y).mean(axis=1) / ((x**2).mean(axis=1) * (y**2).mean(axis=1))**0.5
    return cc

def corr_matrix(x):
    #cc = np.corrcoef(x)
    x  = zscore(x,axis=1)
    cc = x @ x.T / x.shape[1]
    cc = cc + np.diag(np.inf*np.ones(cc.shape[0]))
    return cc

def peer_pred_worker(data):
    Xtest, y, npeers, ic, alg = data
    # split again into 50 neuron chunks
    nn = ic.size
    ns = 25
    nc = int(np.ceil(nn / ns))
    ichunks = np.round(np.linspace(0, nn, nc+1)).astype(int)
    cc_pred = np.zeros((npeers.size,),np.float32)
    for c in range(nc):
        ics = ic[ichunks[c]:ichunks[c+1]]
        if alg=='corr':
            dists = -1 * Xtest[ics,:] @ Xtest.T / Xtest.shape[1]
        else:
            dists = embedding_distance(y[ics,:], y, alg)
        dists[np.arange(0,ics.size).astype(int), ics] = np.inf
        nbest = np.argsort(dists, axis=1)
        peers = nbest[:, :npeers[-1]]
        #peer_pred = np.cumsum(Xtest[peers,:], axis=1)
        peer_pred = Xtest[peers,:]
        #print(peer_pred.shape)
        for ip,n in enumerate(npeers):
            cc = corr(Xtest[ics,:], peer_pred[:,:n,:].mean(axis=1))
            cc_pred[ip] += cc.mean()
    cc_pred /= nc
    return cc_pred

def peer_pred(Xtest, y, alg, npeers):
    ''' test prediction from peers in embedding
        npeers are how many peers to use (can be vector)'''
    if npeers is int:
        npeers = np.array([npeers])
    NN = Xtest.shape[0]
    Xtest = zscore(Xtest, axis=1)
    cc_pred = np.zeros((npeers.size,))
    peer_pred = np.zeros((NN,Xtest.shape[1]))
    nn = 250
    nc = int(np.ceil(NN / nn))
    ichunks = np.round(np.linspace(0, NN, nc+1)).astype(int)
    dsplit = []
    rperm = np.random.permutation(NN)
    ncores = min(10, nc)
    icl = []
    for c in range(ncores):
        ic = np.arange(ichunks[c], ichunks[c+1]).astype(int)
        ic = rperm[ic]
        icl.append(ic)
        dsplit.append([Xtest, y, npeers, ic, alg])
    with Pool(ncores) as p:
        results = p.map(peer_pred_worker, dsplit)
    for c in range(ncores):
        cc_pred += results[c]
    cc_pred /= ncores
    return cc_pred

def dist_corr_worker(data):
    Xtest, y, ncomp, ic, alg = data
    # don't need to split again?
    nn = ic.size
    ns = 250
    nc = int(np.ceil(nn / ns))
    ichunks = np.round(np.linspace(0, nn, nc+1)).astype(int)
    csort = np.zeros((ncomp,),np.float32)
    for c in range(nc):
        ics = ic[ichunks[c]:ichunks[c+1]]
        nr = np.arange(0,ics.size).astype(int)
        if alg=='corr':
            dists = -1 * Xtest[ics,:] @ Xtest.T / Xtest.shape[1]
        else:
            dists = embedding_distance(y[ics,:], y, alg)
        dists[nr, ics] = np.inf
        nbest = np.argsort(dists, axis=1)
        cmat = Xtest[ics,:] @ Xtest.T / Xtest.shape[1]
        cmat_sort = cmat[np.tile(nr[:,np.newaxis], (1,ncomp)), nbest[:, :ncomp]]
        csort += cmat_sort.mean(axis=0)
    csort /= nc
    return csort

def dist_corr(Xtest, y, alg):
    ''' sort neurons by distance from each other and compute correlations '''
    NN = Xtest.shape[0]
    Xtest = zscore(Xtest, axis=1)
    ncomp = min(10000, Xtest.shape[0])
    csort = np.zeros((ncomp,))
    peer_pred = np.zeros((NN,Xtest.shape[1]))
    nn = 250
    nc = int(np.ceil(NN / nn))
    ichunks = np.round(np.linspace(0, NN, nc+1)).astype(int)
    dsplit = []
    rperm = np.random.permutation(NN)
    ncores = min(20, nc)
    icl = []
    for c in range(ncores):
        ic = np.arange(ichunks[c], ichunks[c+1]).astype(int)
        ic = rperm[ic]
        icl.append(ic)
        dsplit.append([Xtest, y, ncomp, ic, alg])
    with Pool(ncores) as p:
        results = p.map(dist_corr_worker, dsplit)
    for c in range(ncores):
        csort += results[c]
    csort /= ncores
    return csort

def global_corr_worker(data):
    Xtest, y, ic, alg = data
    dists = embedding_distance(y[ic,:], y, alg)
    dists[np.arange(0,ic.size).astype(int), ic] = np.inf
    cc = Xtest[ic,:] @ Xtest.T / X.shape[1]
    cc[np.arange(0,ic.size).astype(int), ic] = np.inf
    cc_pred = spearman(cc, dists, True)
    return cc_pred

def spearman(x,y,remove_diag=False):
    ''' spearman correlation of x[i,:] with y[i,:] for all i '''
    # rank correlations
    x = np.argsort(x, axis=1)
    y = np.argsort(y, axis=1)
    # remove diagonals
    if remove_diag:
        x[~np.eye(x.shape[0],dtype=bool)].reshape(x.shape[0],-1)
        y[~np.eye(x.shape[0],dtype=bool)].reshape(x.shape[0],-1)
    cc = corr(x,y)
    return cc
