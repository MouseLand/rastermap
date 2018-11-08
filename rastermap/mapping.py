from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
import math
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
        q1  = xs[n,:][:,np.newaxis] - xs[n,:]
        d0  += dwrap(q1, nclust0)**2
        q1  = xn[n,:][:,np.newaxis] - xs[n,:]
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

class rastermap:
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
                 upsamp=25, sig_upsamp=1.0, sig_Y=1.0,
                 sig_anneal=np.concatenate((np.linspace(6,1,60), 1*np.ones(40)), axis=0),
                 #ncomps_anneal=np.concatenate((np.linspace(1,5,60), 5*np.ones(40)), axis=0),
                 ncomps_anneal=np.concatenate((np.linspace(1,7,60), 7*np.ones(40)), axis=0),
                 init='pca'):
        self.n_components = n_components
        self.n_X = n_X
        self.n_Y = n_Y
        self.iPC = iPC
        self.upsamp = upsamp
        self.sig_upsamp = sig_upsamp
        self.sig_Y = sig_Y
        self.sig_anneal = sig_anneal.astype(np.float32)
        self.ncomps_anneal = ncomps_anneal.astype(np.int32)
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
        dims = self.n_components
        if hasattr(self, 'isort1'):
            if X.shape[1] == self.v.shape[0]:
                # reduce dimensionality of X
                X = X @ self.v
                nclust = self.n_X
                upsamp = int(self.upsamp ** 0.5)
                Km = upsampled_kernel(nclust**2, self.sig_upsamp, upsamp, dims)

                smap = (X @ self.A.T) @ self.S
                nmap = np.sum(self.S * ((self.A @ self.A.T) @ self.S), axis=0)
                cmap = np.maximum(0, smap)**2 / nmap

                iclustup = np.argmax(np.sqrt(cmap) @ Km.T, axis=1)
                n   = nclust * upsamp
                iclustx,iclusty = np.unravel_index(iclustup, (n,n))
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

    def _create_2D_basis(self, K, nclust):
        Kx = self._create_1D_basis(K, nclust)
        Kx = Kx.T
        S = np.zeros((2*K+1, 2*K+1, nclust, nclust), np.float32)
        f = np.zeros((2*K+1, 2*K+1))
        for kx in range(2*K+1):
            for ky in range(2*K+1):
                S[ky,kx,:,:] = np.outer(Kx[:,ky], Kx[:,kx])
                f[ky,kx] = ky + kx
        isort = np.argsort(f.flatten())
        S = np.reshape(S, ((2*K+1)**2, nclust**2))
        S = S[isort,:]
        #S = S / (1 + np.arange(S.shape[0]))[:, np.newaxis]**.5
        #S = S[int(((2*K+1)**2-1)/2):, :]
        return S

    def _create_3D_basis(self, K, nclust):
        Kx = self._create_1D_basis(K, nclust)
        Kx = Kx.T
        S = np.zeros((2*K+1, 2*K+1, 2*K+1, nclust, nclust, nclust), np.float32)
        for kx in range(2*K+1):
            for ky in range(2*K+1):
                for kz in range(2*K+1):
                    S[kz,ky,kx,:,:,:] = np.outer(Kx[:,ky], Kx[:,kx])[:,:, np.newaxis] * Kx[:, kz]
        S = np.reshape(S, ((2*K+1)**3, nclust**3))
        return S

    def _create_1D_basis(self, K, nclust):
        xs = np.arange(0,nclust)
        Kx = np.ones((nclust, 2*K+1), 'float32')
        for k in range(K):
            Kx[:,2*k+1] = np.sin(2*math.pi * (xs+0.5) *  (1+k)/nclust)
            Kx[:,2*k+2] = np.cos(2*math.pi * (xs+0.5) *  (1+k)/nclust)
        Kx /= np.sum(Kx**2, axis=0)**.5
        S = Kx.T
        return S

    def _map(self, X, dims, nclust, u):
        NN,nPC = X.shape
        upsamp = self.upsamp
        # initialize 1D clusters as nodes of 1st PC
        if dims==1:
            xid = np.floor(nclust * np.argsort(u).astype(np.float32)/NN)
        elif dims==2:
            #nclust = int(nclust * upsamp**.5)
            xc = np.floor(nclust * np.argsort(u[:,0]).astype(np.float32)/NN)
            yc = np.floor(nclust * np.argsort(u[:,1]).astype(np.float32)/NN)
            xid = yc + xc * nclust
            upsamp = int(self.upsamp ** 0.5)
        elif dims==3:
            xc = np.floor(nclust * np.argsort(u[:,0]).astype(np.float32)/NN)
            yc = np.floor(nclust * np.argsort(u[:,1]).astype(np.float32)/NN)
            zc = np.floor(nclust * np.argsort(u[:,2]).astype(np.float32)/NN)
            xid = zc + nclust * (yc + xc * nclust)
        xid = xid.astype('int').flatten()

        n_comps = 9
        if dims==2:
            SALL = self._create_2D_basis(n_comps, nclust)
        elif dims==1:
            SALL = self._create_1D_basis(n_comps, nclust)
        elif dims==3:
            SALL = self._create_3D_basis(n_comps, nclust)

        #X = (X - X.mean(axis=1)[:,np.newaxis]) / X.var(axis=1)[:,np.newaxis] ** 0.5
        tic = time.time()
        xmap = np.ones(NN, np.float32)
        ncomps_anneal = np.concatenate((np.linspace(4,SALL.shape[0],60), SALL.shape[0]*np.ones(40)), axis=0).astype('int')
        lam = .00
        xnorm = (X**2).sum(axis=1)[:,np.newaxis]
        if dims==2:
            Km = upsampled_kernel(nclust**2, self.sig_upsamp, upsamp, dims)

        for t,nc in enumerate(ncomps_anneal):
            # get basis functions
            S = SALL[:nc, :]
            S0 = S[:, xid]  #* xmap
            eweights = (S0.T @ S)
            #print(np.mean(np.diag(xtx)))
            #xty = (S0 @ X) /NN
            #A = np.linalg.solve(xtx + lam * np.eye(nc),  xty)
            A = S0 @ X
            #A /= np.sum(A**2, axis=1)[:, np.newaxis]**.5

            vnorm   = np.sum(S * ((A @ A.T) @ S), axis=0)[np.newaxis,:]
            cv      = (X @ A.T) @ S
            vnorm   = vnorm + xnorm * eweights**2 - 2*eweights * cv
            cv      = cv - xnorm * eweights
            cmap    = np.maximum(0., cv)**2 / vnorm

            #smap = (X @ A.T) @ S - xnorm * eweights
            #nmap = np.sum(S * ((A @ A.T) @ S), axis=0)
            #xmap = np.maximum(0, smap[np.arange(NN), xid]) / nmap[xid]
            # cmap = np.maximum(0, smap)**2 / nmap

            xid = np.argmax(cmap, axis=1)
            xmap = np.maximum(0, cv[np.arange(NN), xid]) / vnorm[np.arange(NN), xid]
            cost = np.amax(cmap, axis=1)
            #print(np.amax(xid))
            if t%10==0 or t==self.sig_anneal.size-1:
                print('iter %d; 0/1-clusts: %d; self-corr: %4.4f; time: %2.2f s'%(t, np.isnan(cost).sum(), np.nanmean(cost), time.time()-tic))

        iclustup = np.argmax(np.sqrt(cmap) @ Km.T, axis=1)
        isort = np.argsort(iclustup)
        if dims==2:
            n = nclust * upsamp
            iclustx,iclusty = np.unravel_index(iclustup, (n,n))
            iclustup = np.vstack((iclustx,iclusty)).T
            print(iclustup.shape)

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
