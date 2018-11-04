from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
import numpy as np
import time

def upsampled_kernel(nclust, sig, upsamp, dims):
    if dims==2:
        nclust0 = int(nclust**0.5)
    else:
        nclust0 = nclust
    xs = np.arange(0,nclust0)
    xn = np.linspace(0, nclust0-1, nclust0 * upsamp)
    if dims==2:
        xs,ys = np.meshgrid(xs,xs)
        xs = np.vstack((xs.flatten(),ys.flatten()))
        xn,yn = np.meshgrid(xn,xn)
        xn = np.vstack((xn.flatten(),yn.flatten()))
    else:
        xs = xs[np.newaxis,:]
        xn = xn[np.newaxis,:]
    d0 = np.zeros((nclust,nclust),np.float32)
    d1 = np.zeros((xn.shape[1],nclust),np.float32)
    for n in range(dims):
        d0 += (xs[n,:][:,np.newaxis] - xs[n,:][np.newaxis,:])**2
        d1 += (xn[n,:][:,np.newaxis] - xs[n,:][np.newaxis,:])**2
    K0 = np.exp(-1*d0/sig)
    K1 = np.exp(-1*d1/sig)
    Km = K1 @ np.linalg.inv(K0 + 0.001 * np.eye(nclust))
    return Km

def map(S, usort, ops=None, u=None, sv=None):
    if ops is None:
        ops = {'nclust': 30, # number of clusters
               'iPC': np.arange(0,200).astype(np.int32), # number of PCs to use
               'upsamp': 100, # upsampling factor for embedding position
               'sigUp': 1, # standard deviation for upsampling
               'dims': 1
               }

    S = S - S.mean(axis=1)[:,np.newaxis]
    if (u is None) or (sv is None):
        # compute svd and keep iPC's of data
        nmin = min([S.shape[0],S.shape[1]])
        nmin = np.minimum(nmin-1, ops['iPC'].max())
        sv,u = eigsh(S @ S.T, k=nmin)
        v = S.T @ u
    isort = np.argsort(usort).astype(np.int32)
    iPC = ops['iPC']
    iPC = iPC[iPC<sv.size]
    S = u[:,iPC] @ np.diag(sv[iPC])
    NN,nPC = S.shape
    nclust = int(ops['nclust'])
    if ops['dims']==2:
        nclust = nclust**2
    nn = int(np.floor(NN/nclust)) # number of neurons per cluster
    iclust = np.zeros((NN,),np.int32)
    # initial cluster assignments based on 1st PC weights
    iclust[isort] = np.floor(np.arange(0,NN)/nn).astype(np.int32)
    iclust[iclust>nclust] = nclust
    # annealing schedule for embedding
    sig_anneal = np.concatenate((np.linspace(nclust/10,1,50),np.ones((50,),np.float32)), axis=0)
    # kernel for upsampling cluster identities
    Km = upsampled_kernel(nclust,ops['sigUp'],ops['upsamp'],ops['dims'])
    t=0
    for sig in sig_anneal:
        # compute average activity of each cluster
        V = np.zeros((nPC,nclust), np.float32)
        for j in range(0,nclust):
            iin = iclust==j
            V[:,j] = S[iin,:].sum(axis=0)
        if ops['dims']==2:
            V = np.reshape(V,(nPC,int(nclust**0.5),int(nclust**0.5)))
            V = gaussian_filter(V,[0,sig,sig],mode='wrap')
            V = np.reshape(V,(nPC,nclust))
        else:
            V = gaussian_filter1d(V,sig,axis=1,mode='wrap') # smooth activity across clusters
        V /= ((V**2).sum(axis=0)[np.newaxis,:])**0.5 # normalize columns to unit norm
        cv = S @ V # reproject onto activity across neurons
        # recompute best clusters
        iclust = np.argmax(cv, axis=1)
        cmax = np.amax(cv, axis=1)
        print(np.sum(cmax**2))
        t+=1

    iclustup = np.argmax(cv @ Km.T, axis=1)
    isort = np.argsort(iclustup)
    if ops['dims']==2:
        n = ops['nclust'] * ops['upsamp']
        iclustx,iclusty = np.unravel_index(iclustup, (n,n))
        iclustup = np.vstack((iclustx,iclusty)).T
    return isort,iclustup

def main(S,ops=None,u=None,sv=None,v=None):
    if u is None:
        nmin = min([S.shape[0],S.shape[1]])
        nmin = np.minimum(nmin-1, ops['iPC'].max())
        sv,u = eigsh(S @ S.T, k=nmin)
        sv = sv**0.5
        v = S.T @ u
    isort2,iclust2 = map(S.T,ops,v,sv)
    Sm = S - S.mean(axis=1)[:,np.newaxis]
    Sm = gaussian_filter1d(Sm,3,axis=1)
    isort1,iclust1 = map(Sm,ops,u,sv)
    return isort1,isort2
