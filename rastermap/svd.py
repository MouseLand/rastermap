"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from sklearn.decomposition import TruncatedSVD
import numpy as np
from tqdm import trange 
from .utils import bin1d

def subsampled_mean(X, n_mean=1000):
    n_frames = X.shape[0]
    n_mean = min(n_mean, n_frames)
    # load in chunks of up to 100 frames (for speed)
    nt = 100
    n_batches = int(np.floor(n_mean / nt))
    chunk_len = n_frames // n_batches
    avgframe = np.zeros(X.shape[1:], dtype='float32')
    stdframe = np.zeros(X.shape[1:], dtype='float32')
    for n in trange(n_batches):
        Xb = X[n*chunk_len : n*chunk_len + nt].astype('float32')
        avgXb = Xb.mean(axis=0)
        avgframe += avgXb
        stdframe += ((Xb - avgXb[np.newaxis,...])**2).mean(axis=0)
    avgframe /= n_batches
    stdframe /= n_batches

    return avgframe, stdframe

def SVD(X, n_components=250, return_USV=False, transpose=False):
    nmin = np.min(X.shape)
    nmin = min(nmin, n_components)
    
    Xt = X.T if transpose else X
    U = TruncatedSVD(n_components=nmin, 
                     random_state=0).fit_transform(Xt)
    
    if transpose:
        sv = (U**2).sum(axis=0)**0.5
        U /= sv
        V = (X @ U) / sv
        if return_USV:
            return V, sv, U
        else:
            return V
    else:
        if return_USV:
            sv = (U**2).sum(axis=0)**0.5
            U /= sv
            V = (X.T @ U) / sv
            return U, sv, V
        else:
            return U

def subsampled_SVD(X, n_components=500, n_mean=1000, 
                   n_svd=15000, batch_size=1000, exclude=2):
    """ X is frames by voxels / pixels """
    avgframe, stdframe = subsampled_mean(X)
    if exclude > 0:
        smin, smax = np.percentile(stdframe,1), np.percentile(stdframe,99)
        cutoff = np.linspace(smin, smax, 100//exclude + 1)[1]
        exclude = (stdframe < cutoff).flatten()
        print(f'{exclude.sum()} voxels excluded')
    else:
        exclude = np.zeros(avgframe.size, 'bool')

    n_voxels = np.prod(X.shape[1:])
    n_frames = X.shape[0]
    batch_size = min(batch_size, n_frames)
    n_batches = int(min(np.floor(n_svd / batch_size), np.floor(n_frames / batch_size)))
    chunk_len = n_frames // n_batches
    nc = int(250)  # <- how many PCs to keep in each chunk
    nc = min(nc, batch_size - 1)
    if n_batches == 1:
        nc = min(n_components, batch_size - 1)
    n_components = min(nc*n_batches, n_components)

    U = np.zeros(((~exclude).sum(), nc*n_batches), 'float32')
    avgframe_f = avgframe.flatten()[~exclude][np.newaxis,...]
    for n in trange(n_batches):
        Xb = X[n*chunk_len : n*chunk_len + batch_size]
        Xb = Xb.reshape(Xb.shape[0], -1)
        if exclude.sum()>0:
            Xb = Xb[:,~exclude]
        Xb = Xb.copy().astype('float32')
        Xb -= avgframe_f
        Xb = Xb.reshape(Xb.shape[0], -1)
        Ub = SVD(Xb.T, n_components=nc)
        U[:, n*nc : (n+1)*nc] = Ub
    
    if U.shape[-1] > n_components:
        U = SVD(U, n_components=n_components)

    Sv = (U**2).sum(axis=0)**0.5
    U /= Sv

    n_components = U.shape[-1]
    V = np.zeros((n_frames, n_components))
    n_batches = int(np.ceil(n_frames / batch_size))
    for n in trange(n_batches):
        Xb = X[n*batch_size : (n+1)*batch_size]
        Xb = Xb.reshape(Xb.shape[0], -1)
        if exclude.sum()>0:
            Xb = Xb[:,~exclude]
        Xb = Xb.copy().astype('float32')
        Xb -= avgframe_f
        Vb = Xb.reshape(Xb.shape[0],-1) @ U
        V[n*batch_size : (n+1)*batch_size] = Vb

    if exclude.sum()>0:
        Uf = np.nan * np.zeros((n_voxels, n_components), 'float32')
        Uf[~exclude] = U
        U = Uf

    U = U.reshape(*X.shape[1:], U.shape[-1])

    return U, Sv, V