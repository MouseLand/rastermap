from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
import numpy as np
import time

def svdecon(X, k=100):
    NN, NT = X.shape
    if NN>NT:
        COV = (X.T @ X)/NT
    else:
        COV = (X @ X.T)/NN
    if k==0:
        k = np.minimum(COV.shape) - 1
    Sv, U = eigsh(COV, k = k)
    U, Sv = np.real(U), np.abs(Sv)
    U, Sv = U[:, ::-1], Sv[::-1]**.5
    if NN>NT:
        V = U
        U = X @ V
        U = U/(U**2).sum(axis=0)**.5
    else:
        V = (U.T @ X).T
        V = V/(V**2).sum(axis=0)**.5
    return U, Sv, V

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

class RMAP:
    """rastermap embedding algorithm
    Rastermap first takes the specified PCs of the data, and then embeds them into
    n_X clusters. It returns upsampled cluster identities (n_X*upsamp).
    Clusters are also computed across Y (n_Y) and smoothed, to help with fitting.

    data X: n_samples x n_features

    Parameters
    -----------
    n_components : int, optional (default: 1)
        dimension of the embedding space
    n_X : int, optional (default: 30)
        number of clusters in X
    n_Y : int, optional (default: 100)
        number of clusters in Y: will be used to smooth data before sorting in X
    iPC : nparray, int, optional (default: 0-199)
        which PCs to use during optimization
    upsamp : int, optional (default: 25)
        embedding is upsampled in last iteration using kriging interpolation
    sig_upsamp : float, optional (default: 1.0)
        stddev of Gaussian in kriging interpolation for upsampled estimation
    sig_Y : float, optional (default: 3.0)
        stddev of Gaussian smoothing in Y before sorting in X
    sig_anneal: 1D float array, optional (default: starts at 6.0, decreases to 1.0)
        stddev of Gaussian smoothing of clusters, changes across iterations
        default is 50 iterations (last 20 at 1.0)
    init : initialization of algorithm (default: 'pca')
        can use 'pca', 'random', or a matrix n_samples x n_components

    Attributes
    ----------
    embedding : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    u,sv,v : singular value decomposition of data S, potentially with smoothing
    isort1 : sorting along first dimension of matrix
    isort2 : sorting along second dimension of matrix (if n_Y > 0)

    """
    def __init__(self, n_components=1, n_X=30, n_Y=100,
                 iPC=np.arange(0,200).astype(np.int32),
                 upsamp=25, sig_upsamp=1.0, sig_Y=3.0,
                 sig_anneal=np.concatenate((np.linspace(6,1,30), 1*np.ones(20)), axis=0),
                 init='pca'):
        self.n_components = n_components
        self.n_X = n_X
        self.n_Y = n_Y
        self.iPC = iPC
        self.upsamp = upsamp
        self.sig_upsamp = sig_upsamp
        self.sig_Y = sig_Y
        self.sig_anneal = sig_anneal.astype(np.float32)
        self.init = init

    def fit(self, X, u=None, sv=None, v=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : Ignored
        """
        X -= X.mean(axis=1)[:,np.newaxis]
        if (u is None) or (sv is None) or (v is None):
            # compute svd and keep iPC's of data
            nmin = min([X.shape[0],X.shape[1]])
            nmin = np.minimum(nmin-1, self.iPC.max()+1)
            u,sv,v = svdecon(X, k=nmin)
        iPC = self.iPC
        iPC = iPC[iPC<sv.size]
        self.iPC = iPC

        # first smooth in Y (if n_Y > 0)
        # this will be a 1-D fit
        isort2 = []
        if self.n_Y > 0:
            isort2, iclustup = self._map(v[:,iPC] @ np.diag(sv[iPC]), 1, self.n_Y, v[:,0])
            X = gaussian_filter1d(X[:, isort2], self.sig_Y, axis=1)
            X -= X.mean(axis=1)[:,np.newaxis]
            u,sv,v = svdecon(X, k=nmin)

        Xlowd = u[:,iPC] @ np.diag(sv[iPC])
        self.u = u
        self.sv = sv
        self.v = v
        if self.init=='pca':
            init_sort = u[:,:self.n_components]
        elif self.init=='random':
            init_sort = np.random.permutation(X.shape[0])[:,np.newaxis]
            for j in range(1,self.n_components):
                init_sort = np.concatenate((init_sort, np.random.permutation(X.shape[0])[:,np.newaxis]), axis=-1)
        else:
            if init.shape[0] == X.shape[0] and init.shape[1] >= self.n_components:
                init_sort = init[:, :self.n_components]
            else:
                init_sort = u[:,:self.n_components]

        # now sort in X
        isort1, iclustup = self._map(Xlowd, self.n_components, self.n_X, init_sort)
        self.isort2 = isort2
        self.isort1 = isort1
        self.embedding = iclustup
        return self

    def fit_transform(self, X, u=None, sv=None, v=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self.fit(X, u, sv, v)
        return self.embedding

    def transform(self, X):
        """ if already fit, can add new points and see where they fall"""
        iclustup = []
        if hasattr(self, 'isort1'):
            if X.shape[1] == self.v.shape[0]:
                # reduce dimensionality of X
                NN,nPC = self.X.shape
                X = X @ self.v
                # smooth and sort in order of X
                if len(self.isort2) > 0:
                    X = gaussian_filter1d(X[:, self.isort2], self.sig_Y, axis=1)
                nclust0 = self.n_X * self.upsamp
                nclust = (self.n_X * self.upsamp) ** 2
                # create eweights from upsampled embedding
                ys,xs = np.meshgrid(np.arange(0, nclust0),
                                    np.arange(0, nclust0))
                xc = self.iclustup[:,0]
                xc = xc[:, np.newaxis, np.newaxis]
                d = dwrap(xs - xc, nclust0)**2
                if self.n_components==2:
                    yc = self.iclustup[:,1]
                    yc = yc[:, np.newaxis, np.newaxis]
                    d += dwrap(ys - yc, nclust0)**2
                eweights = np.exp(-d/(2**2))
                eweights = np.reshape(eweights, (NN, nclust))
                V = self.X.T @ eweights
                vnorm = (V**2).sum(axis=0)[np.newaxis,:]
                cv = X @ V
                print(cv.shape)
                cv = np.maximum(0., cv)**2 / vnorm
                iclustup = np.argmax(cv, axis=1)
                if self.n_components==2:
                    iclustx,iclusty = np.unravel_index(iclustup, (nclust0, nclust0))
                    iclustup = np.vstack((iclustx,iclusty)).T
            else:
                print('ERROR: new points do not have as many features as original data')
        else:
            print('ERROR: need to fit model first before you can embed new points')
        return iclustup

    def _map(self, X, dims, nclust, u):
        NN,nPC = X.shape
        nn = int(np.floor(NN/nclust)) # number of neurons per cluster
        nclust0 = nclust
        # initialize 2D clusters as nodes of 1st two PCs
        if dims==2:
            xc = nclust0 * np.argsort(u[:,0]).astype(np.float32)/NN
            yc = nclust0 * np.argsort(u[:,1]).astype(np.float32)/NN
            ys,xs = np.meshgrid(np.arange(0,nclust0), np.arange(0,nclust0))
            nclust = nclust**2
            upsamp = int(self.upsamp ** 0.5)
        # initialize 1D clusters as nodes of 1st PC
        else:
            xc = nclust0 * np.argsort(u).astype(np.int32) / NN
            xs = np.arange(0,nclust0)
            upsamp = self.upsamp
        # kernel for upsampling cluster identities
        Km = upsampled_kernel(nclust, self.sig_upsamp, upsamp, dims)
        nnorm = (X**2).sum(axis=1)[:,np.newaxis]
        Xnorm = (X - X.mean(axis=1)[:,np.newaxis]) / X.var(axis=1)[:,np.newaxis] ** 0.5
        tic = time.time()
        for t,sig in enumerate(self.sig_anneal):
            # compute average activity of each cluster
            V = np.zeros((nPC,nclust), np.float32)
            xc = xc[:, np.newaxis, np.newaxis]
            d = dwrap(xs - xc, nclust0)**2
            if dims==2:
                yc = yc[:, np.newaxis, np.newaxis]
                d+= dwrap(ys - yc, nclust0)**2
            eweights = np.exp(-d/(2*sig**2))
            eweights = np.reshape(eweights, (NN, nclust))

            V = X.T @ eweights
            vnorm = (V**2).sum(axis=0)[np.newaxis,:] # normalize columns to unit norm
            # reproject onto activity across neurons (subtract self-proj)
            cv = X @ V - nnorm * eweights
            vnorm = vnorm + nnorm * eweights**2 - 2*eweights * cv
            cv = np.maximum(0., cv)**2 / vnorm
            # best clusters
            iclust = np.argmax(cv, axis=1)
            # cost function: optimize self-correlation of clusters
            V = np.zeros((nclust,nPC), np.float32)
            nc = np.zeros(nclust, np.int32)
            # summed activity of neurons in each cluster
            for j in range(nclust):
                V[j,:] = X[iclust==j, :].sum(axis=0)
                nc[j] = (iclust==j).sum()
            # ignore divide by zero for clusters with 0 or 1 entry
            with np.errstate(divide='ignore', invalid='ignore'):
                Vr = (V[iclust,:] - X) / (nc[iclust][:,np.newaxis] - 1)
                Vr -= Vr.mean(axis=1)[:, np.newaxis]
                Vr /= Vr.var(axis=1)[:, np.newaxis] ** 0.5
            cc = (Xnorm * Vr).mean(axis=1)
            if t%10==0 or t==self.sig_anneal.size-1:
                print('iter %d; 0/1-clusts: %d; self-corr: %4.4f; time: %2.2f s'%(t, np.isnan(cc).sum(), np.nanmean(cc), time.time()-tic))
            # 2D positions
            if dims==2:
                xc,yc = np.unravel_index(iclust, (nclust0,nclust0))
            else:
                xc = iclust

        iclustup = np.argmax(np.sqrt(cv) @ Km.T, axis=1)
        isort = np.argsort(iclustup)
        if dims==2:
            n = nclust0 * upsamp
            iclustx,iclusty = np.unravel_index(iclustup, (n,n))
            iclustup = np.vstack((iclustx,iclusty)).T
        return isort, iclustup

def main(X,ops=None,u=None,sv=None,v=None):
    if u is None:
        nmin = min([X.shape[0],X.shape[1]])
        nmin = np.minimum(nmin-1, ops['iPC'].max())
        sv,u = eigsh(X @ X.T, k=nmin)
        sv = sv**0.5
        v = X.T @ u
    isort2,iclust2 = _map(X.T,ops,v,sv)
    Xm = X - X.mean(axis=1)[:,np.newaxis]
    Xm = gaussian_filter1d(Xm,3,axis=1)
    isort1,iclust1 = _map(Xm,ops,u,sv)
    return isort1,isort2
