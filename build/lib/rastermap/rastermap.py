from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
import numpy as np
import time

def upsampled_kernel(nclust, sig, upsamp):
    xs = np.arange(0,nclust)
    xn = np.linspace(0, nclust-1, nclust * upsamp)
    d0 = (xs[:,np.newaxis] - xs[np.newaxis,:])**2;
    d1 = (xn[:,np.newaxis] - xs[np.newaxis,:])**2;
    K0 = np.exp(-1*d0/sig)
    K1 = np.exp(-1*d1/sig)
    Km = K1 @ np.linalg.inv(K0 + 0.001 * np.eye(nclust));
    return Km

def map(S, ops=None, u=None, sv=None):
    if ops is None:
        ops = {'nclust': 30, # number of clusters
               'iPC': np.arange(0,200).astype(np.int32), # number of PCs to use
               'upsamp': 100, # upsampling factor for embedding position
               'sigUp': 1, # standard deviation for upsampling
               'equal': False # whether or not clusters should be of equal size
               }

    S = S - S.mean(axis=1)[:,np.newaxis]
    if (u is None) or (sv is None):
        # compute svd and keep iPC's of data
        sv,u = eigsh(S @ S.T, k=200)
        v = S.T @ u
    isort = np.argsort(u[:,0]).astype(np.int32)

    iPC = ops['iPC']
    iPC = iPC[iPC<sv.size]
    S = u[:,iPC] @ np.diag(sv[iPC])
    NN,nPC = S.shape
    nclust = int(ops['nclust'])
    nn = int(np.floor(NN/nclust)) # number of neurons per cluster
    iclust = np.zeros((NN,),np.int32)
    # initial cluster assignments based on 1st PC weights
    iclust[isort] = np.floor(np.arange(0,NN)/nn).astype(np.int32)
    iclust[iclust>nclust] = nclust
    # annealing schedule for embedding
    sig_anneal = np.concatenate((np.linspace(nclust/10,1,50),np.ones((50,),np.float32)), axis=0)
    # kernel for upsampling cluster identities
    Km = upsampled_kernel(nclust,ops['sigUp'],ops['upsamp'])
    t=0
    for sig in sig_anneal:
        # compute average activity of each cluster
        if ops['equal'] and t>0:
            iclustup = np.argmax(cv @ Km.T, axis=1)
            isort = np.argsort(iclustup)
            V = S[isort,:]
            V = np.reshape(V[:nn*nclust,:], (nn,nclust,V.shape[1])).sum(axis=0)
            V = V.T
        else:
            V = np.zeros((nPC,nclust), np.float32)
            for j in range(0,nclust):
                iin = iclust==j
                V[:,j] = S[iin,:].sum(axis=0)
        V = gaussian_filter1d(V,sig,axis=1,mode='reflect') # smooth activity across clusters
        V /= ((V**2).sum(axis=0)[np.newaxis,:])**0.5 # normalize columns to unit norm
        cv = S @ V # reproject onto activity across neurons
        # recompute best clusters
        iclust = np.argmax(cv, axis=1)
        cmax = np.amax(cv, axis=1)
        t+=1

    iclustup = np.argmax(cv @ Km.T, axis=1)
    isort = np.argsort(iclustup)
    return isort

def main(S,ops=None,u=None,sv=None,v=None):
    if u is None:
        sv,u = eigsh(S @ S.T, k=200)
        sv = sv**0.5
        v = S.T @ u
    isort2 = map(S.T,ops,v,sv)
    Sm = S - S.mean(axis=1)[:,np.newaxis]
    Sm = gaussian_filter1d(Sm,3,axis=1)
    isort1 = map(Sm,ops,u,sv)
    return isort1,isort2

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
