from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
import math
import numpy as np
import time

def create_ND_basis(dims, nclust, K):
    # recursively call this function until we fill out S
    if dims==1:
        xs = np.arange(0,nclust)
        S = np.ones((K, nclust), 'float32')
        for k in range(K):
           S[k, :] = np.sin(math.pi + (k+1)%2 * math.pi/2 + 2*math.pi/nclust * (xs+0.5) * int((1+k)/2))
        S /= np.sum(S**2, axis = 1)[:, np.newaxis]**.5
        fxx = np.arange(K)
    else:
        S0, fy = create_ND_basis(dims-1, nclust, K)
        Kx, fx = create_ND_basis(1, nclust, K)
        S = np.zeros((S0.shape[0], K, S0.shape[1], nclust), np.float64)
        fxx = np.zeros((S0.shape[0], K))
        for kx in range(K):
            for ky in range(S0.shape[0]):
                S[ky,kx,:,:] = np.outer(S0[ky, :], Kx[kx, :])
                fxx[ky,kx] = fy[ky] + fx[kx]
        S = np.reshape(S, (K*S0.shape[0], nclust*S0.shape[1]))
    fxx = fxx.flatten()
    ix = np.argsort(fxx)
    S = S[ix, :]
    fxx = fxx[ix]
    return S, fxx

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

def upsample(cmap, nclust, Km, upsamp, flag):
    N = cmap.shape[0]
    xid = np.argmax(cmap, axis=1)
    iclustup = xid
    iclustx,iclusty = np.unravel_index(iclustup, (nclust,nclust))
    cmap = np.reshape(cmap, (N, nclust, nclust))
    ys, xs = np.meshgrid(np.arange(-2,3), np.arange(-2,3))
    iclustY = iclusty[:, np.newaxis, np.newaxis] + ys
    iclustX = iclustx[:, np.newaxis, np.newaxis] + xs

    iclustX = iclustX%nclust
    iclustY = iclustY%nclust

    iN = np.arange(N)[:, np.newaxis, np.newaxis] + np.zeros((5,5))
    C0 = cmap[iN.flatten().astype('int'), iclustX.flatten().astype('int'), iclustY.flatten().astype('int')]
    C0 = np.reshape(C0, (N, 5*5))
    mu = C0.mean(axis=1)
    C0 -= mu[:, np.newaxis]

    upC = C0 @ Km.T
    xid = np.argmax(upC, axis=1)
    cmax = np.amax(upC, axis=1) + mu
    iclustx0,iclusty0 = np.unravel_index(xid, (4*upsamp+1,4*upsamp+1))

    iclustx0 = iclustx0/upsamp - 2.
    iclusty0 = iclusty0/upsamp - 2.

    if flag:
        iclustx = iclustx + iclustx0
        iclusty = iclusty + iclusty0
    return iclustx, iclusty, cmax

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
    xn = np.linspace(0, nclust0-1, (nclust0-1) * upsamp + 1)
    #xn = xn[:-1]
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
        q1  = xs[n,:][:,np.newaxis] - xs[n,:]
        #d0  += dwrap(q1, nclust0)**2
        d0  += q1**2
        q1  = xn[n,:][:,np.newaxis] - xs[n,:]
        #d1  += dwrap(q1, nclust0)**2
        d1  += q1**2

    K0 = np.exp(-d0/sig**2)
    K1 = np.exp(-d1/sig**2)
    Km = K1 @ np.linalg.inv(K0 + .01 * np.eye(nclust))
    return Km

def _getSup(self, iclustx, iclusty, K, nclust, isort, nu):
    NN = len(iclustx)
    S0 = np.ones((K, K, NN), np.float32)

    for kx in range(K):
        for ky in range(K):
            #y = (ky+1)%2 * math.pi/2 + 2*math.pi/nclust * (iclusty+0.5) * int((1+ky)/2)
            #x = (kx+1)%2 * math.pi/2 + 2*math.pi/nclust * (iclustx+0.5) * int((1+kx)/2)
            #S0[ky, kx, :]  = np.sin(x) * np.sin(y) /nu[kx]/nu[ky]
            y = math.pi * (iclusty+0.5) *  ky/nclust
            x = math.pi * (iclustx+0.5) *  kx/nclust
            S0[ky, kx, :]  = np.cos(x) * np.cos(y) /nu[kx]/nu[ky]
    S0 = np.reshape(S0, (K**2, NN))
    S0 = S0[isort,:]
    return S0

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

class Rastermap:
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
    def __init__(self, n_components=2, n_X = -1, n_Y = 0,
                 iPC=np.arange(0,400).astype(np.int32),
                 sig_Y=1.0, init='pca', alpha = 1., K = 1.,
                 mode = 'powerlaw'):

        self.n_components = n_components
        self.alpha = alpha
        self.K     = K
        self.iPC = iPC
        self.sig_Y = sig_Y
        self.init = init
        self.mode = 'powerlaw'
        self.feats_mode = 'pca' # rmap1d
        self.n_Y = n_Y
        if n_X>0:
            self.n_X = n_X
        else:
            self.n_X  = np.ceil(1600**(1/n_components)).astype('int')

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
        dims = self.n_components
        if hasattr(self, 'isort1'):
            if X.shape[1] == self.v.shape[0]:
                # reduce dimensionality of X
                X = X @ self.v
                nclust = self.n_X

                A = self.A
                S = self.S
                AtS = A.T @ S
                vnorm   = np.sum(S * (A @ AtS), axis=0)[np.newaxis,:]
                cv      = X @ AtS
                cmap    = np.maximum(0., cv)**2 / vnorm

                Km = upsampled_kernel(5**2, self.sig_upsamp, 10, dims)
                iclustx, iclusty, cmax = upsample(np.sqrt(cmap), nclust, Km, 10, True)
                iclustup = np.vstack((iclustx,iclusty)).T


#                if dims==2:
#                    nclust = np.sqrt(self.S.shape[1]).astype('int')
#                    iclustx,iclusty = np.unravel_index(iclustup, (nclust,nclust))
#                    iclustup = np.vstack((iclustx,iclusty)).T
#                    self.cmap0 = cmap
#                elif dims==3:
#                    iclustx,iclusty, iclustz = np.unravel_index(iclustup, (nclust,nclust, nclust))
#                    iclustup = np.vstack((iclustx,iclusty, iclustz)).T
            else:
                print('ERROR: new points do not have as many features as original data')
        else:
            print('ERROR: need to fit model first before you can embed new points')
        return iclustup

    def _create_2D_basis0(self, K, nclust):
        xs,ys = np.meshgrid(np.arange(nclust), np.arange(nclust))
        S = np.zeros((nclust, nclust, nclust, nclust), np.float64)
        x0 = np.arange(nclust)
        sig = 1.
        for kx in range(nclust):
            for ky in range(nclust):
                ds = dwrap(xs - x0[kx], nclust)**2 + dwrap(ys - x0[ky], nclust)**2
                S[ky, kx, :, :] = np.exp(-ds**.5/sig)
        S = np.reshape(S, (nclust**2, nclust**2))
        U, Sv, S = svdecon(S, k = K)
        S = S.T
        return S

    def _map(self, X, dims, nclust, u):
        NN,nPC = X.shape
        # initialize 1D clusters as nodes of 1st PC
        iclust = np.zeros((dims, NN))
        xid = np.zeros(NN)
        for j in range(dims):
            iclust[j] = np.floor(nclust * np.argsort(u[:,j]).astype(np.float32)/NN)
            xid = nclust * xid + iclust[j]
        xid = xid.astype('int').flatten()
        SALL, fxx = create_ND_basis(dims, nclust, int(np.ceil(2/3 * nclust)))
        print(SALL.shape)
        tic = time.time()
        ncomps_anneal = np.concatenate((np.linspace(4,SALL.shape[0],30), SALL.shape[0]*np.ones(20)), axis=0).astype('int')
        xnorm = (X**2).sum(axis=1)[:,np.newaxis]
        E = np.zeros(len(ncomps_anneal))
        for t,nc in enumerate(ncomps_anneal):
            # get basis functions
            S = SALL[:nc, :]
            S0 = S[:, xid]
            A  = S0 @ X
            if self.mode == 'powerlaw':
                nA  = np.sum(A**2, axis=1)**.5 * (self.K + np.arange(nc))**(self.alpha/2)
            else:
                nA = np.ones(A.shape[0])
            A        /= nA[:, np.newaxis]
            eweights = (S0.T / nA) @ S
            AtS     = A.T @ S
            vnorm   = np.sum(S * (A @ AtS), axis=0)[np.newaxis,:]
            cv      = X @ AtS
            vnorm   = vnorm + xnorm  * eweights**2 - 2*eweights * cv
            cv      = cv - xnorm * eweights
            cmap    = np.maximum(0., cv)**2 / vnorm
            xid     = np.argmax(cmap, axis=1)
            E[t]    = np.nanmean(np.amax(cmap, axis=1))
            if t%10==0 or t==(ncomps_anneal.size-1):
                print('iter %d; self-corr: %4.4f; time: %2.2f s'%(t, E[t], time.time()-tic))

        self.E = E
        if dims==2:
            Km = upsampled_kernel(5**2, 1.0, 10, dims)
            iclustx, iclusty, cmax = upsample(np.sqrt(cmap), nclust, Km, 10, True)
            isort = np.argsort(xid)
            iclustup = np.vstack((iclustx,iclusty)).T
            print(iclustup.shape)

        #Km = upsampled_kernel(nclust**2, self.sig_upsamp, 5, dims)
        #iclustup = np.argmax(np.sqrt(cmap) @ Km.T, axis=1)
        #n = nclust * 5+1
        #isort = np.argsort(iclustup)
        #iclustx,iclusty = np.unravel_index(iclustup, (n,n))
        #iclustup = np.vstack((iclustx,iclusty)).T

        #if dims==2:
            #nclust = int(nclust * upsamp**.5)
            #S = self._create_2D_basis(n_comps, nclust)
        #elif dims==1:
            #nclust = int(nclust * upsamp)
            #S = self._create_1D_basis(n_comps, nclust)
        #elif dims==3:
            #nclust = int(nclust * upsamp**(1/3))
            #S = self._create_3D_basis(n_comps, nclust)
        #S0 = S[:, xid]  #* xmap
        #A = S0 @ X
        #eweights = (S0.T @ S)
        #vnorm   = np.sum(S * ((A @ A.T) @ S), axis=0)[np.newaxis,:]
        #cv      = (X @ A.T) @ S
        #vnorm   = vnorm + xnorm * eweights**2 - 2*eweights * cv
        #cv      = cv - xnorm * eweights
        #cmap    = np.maximum(0., cv)**2 / vnorm
        #iclustup= np.argmax(cmap, axis=1)

        #smap = (X @ A.T) @ S
        #nmap = np.sum(S * ((A @ A.T) @ S), axis=0)
        #cmap = np.maximum(0, smap)**2 / nmap
        #iclustup = np.argmax(cmap, axis=1)
        #cost = np.amax(cmap, axis=1)
        #print('iter %d; 0/1-clusts: %d; self-corr: %4.4f; time: %2.2f s'%(t, np.isnan(cost).sum(), np.nanmean(cost), time.time()-tic))

        self.S = S
        self.A = A
        self.cmap = cmap
        self.xid = xid
        #isort = np.argsort(iclustup)
        #if dims==2:
            #iclustx,iclusty = np.unravel_index(iclustup, (nclust,nclust))
            #iclustup = np.vstack((iclustx,iclusty)).T
        #elif dims==3:
            #iclustx,iclusty, iclustz = np.unravel_index(iclustup, (nclust,nclust, nclust))
            #iclustup = np.vstack((iclustx,iclusty, iclustz)).T
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
