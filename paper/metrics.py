"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from sklearn.manifold import SpectralEmbedding, Isomap, LocallyLinearEmbedding
import time
from scipy.stats import zscore, spearmanr
from multiprocessing import Pool
from scipy.spatial.distance import pdist
import numpy as np
import scipy
from openTSNE import TSNE, affinity, TSNEEmbedding
from umap import UMAP

def emb_to_idx(emb):
    if emb.ndim==2:
        embs = emb[:,0]
    else:
        embs = emb
    isort = embs.argsort()
    idx = np.zeros_like(isort)
    idx[isort] = np.arange(0, len(isort))
    return idx

def triplet_order(gt, emb):
    """ benchmarking triplet score for embedding with ground truth"""
    if (gt<1).sum() == len(gt):
        idx_gt = emb_to_idx(gt)
        idx_emb = emb_to_idx(emb)
        nn = len(idx_gt)
        correct_triplets = 0
        nrand = nn * 10
        for i in range(nrand):
            i_gt = np.random.choice(nn, size=3, replace=False)
            i_gt = i_gt[i_gt.argsort()]
            triplet_gt = np.array([np.nonzero(idx_gt==k)[0][0] for k in i_gt])
            triplet_emb = idx_emb[triplet_gt]
            if ((triplet_emb[0] < triplet_emb[1] and triplet_emb[1] < triplet_emb[2]) or 
                (triplet_emb[0] > triplet_emb[1] and triplet_emb[1] > triplet_emb[2])):
                correct_triplets += 1
        return correct_triplets / nrand
    else:
        n_modules = int(np.ceil(gt.max()))
        triplet_scores = np.zeros(n_modules)
        for k in range(n_modules): 
            inds = np.floor(gt) == k
            triplet_scores[k] = triplet_order(gt[inds]-k, emb[inds])
        return triplet_scores

def embedding_contamination(gt, emb):
    """ benchmarking contamination score for embedding with ground truth"""
    n_modules = int(np.ceil(gt.max()))
    contamination_scores = np.zeros(n_modules)
    isort = emb.flatten().argsort()
    for k in range(n_modules): 
        imod = np.floor(gt) == k
        in_mod = np.nonzero(imod)[0]
        nn = len(in_mod)
        nrand = nn * 10
        nk = 0
        for i in range(nrand):
            i_gt = in_mod[np.random.choice(nn, size=2, replace=False)]
            i0 = np.nonzero(isort == i_gt[0])[0][0]
            i1 = np.nonzero(isort == i_gt[1])[0][0]
            if np.abs(i1 - i0) > 5:
                i0, i1 = min(i0, i1), max(i0, i1)
                contamination_scores[k] += 1 - imod[isort[i0+1 : i1-1]].mean()
                nk+=1
        contamination_scores[k] /= nk
    return contamination_scores 

def benchmarks(xi_all, embs):
    n_modules = int(np.ceil(xi_all.max()))
    emb_rand = np.random.rand(len(embs[0]), 1).astype("float32")
    contamination_scores = np.zeros((len(embs)+1, n_modules))
    for k, emb in enumerate(embs):
        contamination_scores[k] = embedding_contamination(xi_all, emb)
    contamination_scores[k+1] = embedding_contamination(xi_all, emb_rand)
    
    triplet_scores = np.zeros((len(embs)+1, n_modules))
    for k, emb in enumerate(embs):
        triplet_scores[k] = triplet_order(xi_all, emb)
    triplet_scores[k+1] = triplet_order(xi_all, emb_rand)
    
    return contamination_scores, triplet_scores

def embedding_quality_gt(gt, embs, knn=[10,50,100,200,500]):
    """ benchmarking local and global scores for embedding with ground truth """
    idx_gt = emb_to_idx(gt)[:,np.newaxis]
    mnn = np.zeros((len(embs), len(knn)))
    rho = np.zeros((len(embs),))
    for k,emb in enumerate(embs):
        idx_emb = emb_to_idx(emb)[:,np.newaxis]    
        mnn[k], rho[k] = embedding_quality(idx_gt, idx_emb, knn=knn)
    return mnn, rho

def distance_matrix(Z, n_X=None, wrapping=False, correlation=False):
    if wrapping:
        #n_X = int(np.floor(Z.max() + 1))
        dists = (Z - Z[:, np.newaxis, :]) % n_X
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

def embedding_quality(X, Z, knn=[10,50,100,200,500], subsetsize=2000,
                      wrapping=False, correlation=False, n_X=None):
    """ changed correlation to False and n_X from 0 to None"""
    from sklearn.neighbors import NearestNeighbors
    np.random.seed(101)
    if subsetsize < X.shape[0]:
        subset = np.random.choice(X.shape[0], size=subsetsize, replace=False)
    else:
        subsetsize = X.shape[0]
        subset = slice(0, X.shape[0])
    Xdist = distance_matrix(X[subset], correlation=correlation)
    Zdist = distance_matrix(Z[subset], n_X=n_X, wrapping=wrapping)
    xd = Xdist[np.tril_indices(Xdist.shape[0], -1)]
    zd = Zdist[np.tril_indices(Xdist.shape[0], -1)]
    mnn = []
    if not isinstance(knn, (np.ndarray, list)):
        knn = [knn]
    elif isinstance(knn, np.ndarray):
        knn = list(knn)

    for kni in knn:
        nbrs1 = NearestNeighbors(n_neighbors=kni, metric="precomputed").fit(Xdist)
        ind1 = nbrs1.kneighbors(return_distance=False)

        nbrs2 = NearestNeighbors(n_neighbors=kni, metric="precomputed").fit(Zdist)
        ind2 = nbrs2.kneighbors(return_distance=False)
        
        intersections = 0.0
        for i in range(subsetsize):
            intersections += len(set(ind1[i]) & set(ind2[i]))
        mnn.append(intersections / subsetsize / kni)

    rho = spearmanr(xd, zd).correlation

    return (mnn, rho)

def run_TSNE(U, perplexities=[30], metric="cosine", verbose=False):
    if len(perplexities) > 1:
        affinities_annealing = affinity.PerplexityBasedNN(
                                U,
                                perplexity=perplexities[1],
                                metric=metric,
                                n_jobs=16,
                                random_state=1,
                                verbose=verbose
        )
        embedding = TSNEEmbedding(
                                U[:,:1]*0.0001,
                                affinities_annealing,
                                negative_gradient_method="fft",
                                random_state=1,
                                n_jobs=16,
                                verbose=verbose
                            )
        embedding1 = embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5)
        embedding2 = embedding1.optimize(n_iter=750, exaggeration=1, momentum=0.8)

        affinities_annealing.set_perplexity(perplexities[0])
        embeddingOPENTSNE = embedding2.optimize(n_iter=500, momentum=0.8)
    else:
        tsne = TSNE(
            perplexity=perplexities[0],
            metric=metric,
            n_jobs=16,
            random_state=1,
            verbose=verbose,
            n_components = 1,
            initialization = .0001 * U[:,:1],
        )
        embeddingOPENTSNE = tsne.fit(U)
        
    return embeddingOPENTSNE

def run_UMAP(U, n_neighbors=15, min_dist=0.1, metric="cosine"):
    embeddingUMAP = UMAP(n_components=1, n_neighbors=n_neighbors, random_state=1,
                         min_dist=min_dist, init=U[:,:1], metric=metric).fit_transform(U)
    return embeddingUMAP

def run_LE(U):
    LE = SpectralEmbedding(n_components=1, n_jobs=16, random_state=1).fit(U)
    return LE.embedding_
    
def run_LLE(U, n_neighbors=5):
    LLE = LocallyLinearEmbedding(n_components=1, n_jobs=16, n_neighbors=n_neighbors, random_state=1).fit(U)
    return LLE.embedding_

def run_isomap(U, n_neighbors=5, metric="cosine"):
    IM = Isomap(n_components=1, n_jobs=16, n_neighbors=n_neighbors, 
                metric=metric).fit(U)
    return IM.embedding_

