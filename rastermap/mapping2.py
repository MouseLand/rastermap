from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
import numpy as np
import time


def dwrap(kx,nc):
    '''compute a wrapped distance'''
    q1 = np.mod(kx, nc)
    q2 = np.minimum(q1, nc-q1)
    return q2

def upsampled_kernel(nclust, sig, upsamp, dims):
    if dims==2:
        nclust0 = int(nclust**0.5)
    else:
        nclust0 = nclust
    xs = np.arange(0,nclust0)
    xn = np.linspace(0, nclust0, nclust0 * upsamp + 1)
    xn = xn[:-1]
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
        q1  = xs[n,:][:,np.newaxis] - xs[n,:][np.newaxis,:]
        d0  += dwrap(q1, nclust0)**2
        q1  = xn[n,:][:,np.newaxis] - xs[n,:][np.newaxis,:]
        d1  += dwrap(q1, nclust0)**2

    K0 = np.exp(-d0/sig**2)
    K1 = np.exp(-d1/sig**2)
    Km = K1 @ np.linalg.inv(K0 + 0.01 * np.eye(nclust))
    return Km

def map(S, ops=None, u=None, sv=None):
    if ops is None:
        ops = {'nclust': 30, # number of clusters
               'iPC': np.arange(0,200).astype(np.int32), # number of PCs to use
               'upsamp': 100, # upsampling factor for embedding position
               'sigUp': 1., # standard deviation for upsampling
               'dims': 1
               }

    S = S - S.mean(axis=1)[:,np.newaxis]
    if (u is None) or (sv is None):
        # compute svd and keep iPC's of data
        nmin = min([S.shape[0],S.shape[1]])
        nmin = np.minimum(nmin-1, ops['iPC'].max())
        sv,u = eigsh(S @ S.T, k=nmin)
        sv = sv**0.5
        v = S.T @ u
    iPC = ops['iPC']
    iPC = iPC[iPC<sv.size]
    S = u[:,iPC] @ np.diag(sv[iPC])
    NN,nPC = S.shape
    nclust = int(ops['nclust'])
    nn = int(np.floor(NN/nclust)) # number of neurons per cluster
    nclust0 = nclust
    # initialize 2D clusters
    if ops['dims']==2:
        nclust = ops['nclust']
        xc = nclust0 * np.argsort(u[:,0]).astype(np.float32)/NN
        yc = nclust0 * np.argsort(u[:,1]).astype(np.float32)/NN
        nclust = nclust**2
    # initialize 1D clusters
    else:
        isort = np.argsort(u[:,0]).astype(np.int32)
        iclust = np.zeros((NN,),np.int32)
        iclust[isort] = np.floor(np.arange(0,NN)/nn).astype(np.int32)
        iclust[iclust>nclust] = nclust
    # annealing schedule for embedding
    sig_anneal = np.concatenate((np.linspace(6,1,30),1*np.ones((20,),np.float32)), axis=0)
    # kernel for upsampling cluster identities
    Km = upsampled_kernel(nclust,ops['sigUp'],ops['upsamp'],ops['dims'])
    nnorm = (S**2).sum(axis=1)[:,np.newaxis]
    ys,xs = np.meshgrid(np.arange(0,nclust0), np.arange(0,nclust0))
    #xc, yc = np.unravel_index(iclust, (nclust0,nclust0))
    tic = time.time()
    for t,sig in enumerate(sig_anneal):
        # compute average activity of each cluster
        #xc, yc = np.unravel_index(iclust, (nclust0,nclust0))
        xc = xc[:, np.newaxis, np.newaxis]
        yc = yc[:, np.newaxis, np.newaxis]
        d = dwrap(xs - xc, nclust0)**2 +dwrap(ys - yc, nclust0)**2
        eweights = np.exp(-d/(2*sig**2))
        eweights = np.reshape(eweights, (NN, nclust))
        if ops['dims']==2:
            V = S.T @ eweights
        else:
            V = gaussian_filter1d(V,sig,axis=1,mode='wrap') # smooth activity across clusters
        vnorm = (V**2).sum(axis=0)[np.newaxis,:] # normalize columns to unit norm
        cv = S @ V - nnorm * eweights # reproject onto activity across neurons
        vnorm = vnorm + nnorm * eweights**2 - 2*eweights * cv
        cv = np.maximum(0., cv)**2 / vnorm
        # recompute best clusters
        cvup = cv @ Km.T
        iclustup = np.argmax(cvup, axis=1)
        n = ops['nclust'] * ops['upsamp']
        xc,yc = np.unravel_index(iclustup, (n,n))
        xc = xc/ops['upsamp']
        yc = yc/ops['upsamp']
        cmax = np.amax(cvup, axis=1)
        print('%d %4.4f %2.4f'%(t, np.sum(cmax), time.time()-tic))

    iclustup = np.argmax(np.sqrt(cv) @ Km.T, axis=1)
    isort = np.argsort(iclustup)
    if ops['dims']==2:
        n = ops['nclust']
        n = ops['nclust'] * ops['upsamp']
        iclustx,iclusty = np.unravel_index(iclustup, (n,n))
        iclustup = np.vstack((iclustx,iclusty)).T
    return isort,iclustup,cv

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
