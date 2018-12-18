from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore, skew
import math
import numpy as np
import time


def distances(x, y):
    # x and y are n_components by number of data points
    ds = np.zeros((x.shape[0], y.shape[0]))
    if len(x.shape)==1:
        x = np.reshape(x, (-1, 1))
        y = np.reshape(y, (-1, 1))
    for j in range(x.shape[1]):
        ds += dwrap(x[:,j][:,np.newaxis] - y[:,j], 1.)**2
    ds = np.sqrt(ds)
    return ds

def create_ND_basis(dims, nclust, K, flag=True):
    # recursively call this function until we fill out S
    flag = False
    if dims==1:
        xs = np.arange(0,nclust)
        S = np.ones((K, nclust), 'float32')
        for k in range(K):
            if flag:
                S[k, :] = np.sin(math.pi + (k+1)%2 * math.pi/2 + 2*math.pi/nclust * (xs+0.5) * int((1+k)/2))
            else:
                S[k, :] = np.cos(math.pi/nclust * (xs+0.5) * k)
        S /= np.sum(S**2, axis = 1)[:, np.newaxis]**.5
        fxx = np.floor((np.arange(K)+1)/2).astype('int')
        #fxx = np.arange(K).astype('int')
    else:
        S0, fy = create_ND_basis(dims-1, nclust, K, flag)
        Kx, fx = create_ND_basis(1, nclust, K, flag)
        S = np.zeros((S0.shape[0], K, S0.shape[1], nclust), np.float64)
        fxx = np.zeros((S0.shape[0], K))
        for kx in range(K):
            for ky in range(S0.shape[0]):
                S[ky,kx,:,:] = np.outer(S0[ky, :], Kx[kx, :])
                # fxx[ky,kx] = fy[ky] + fx[kx]
                fxx[ky,kx] = max(fy[ky], fx[kx]) + min(fy[ky], fx[kx])/1000.
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

def upsample(cmap, dims, nclust, upsamp):
    N = cmap.shape[0]
    # first we need the coordinates of the maximum as an array
    nup = 5
    if dims>3:
        nup = 3
    if dims>4:
        upsamp = 4
    if dims>6:
        upsamp = 2
    Km, M1, M0 = upsampled_kernel(nup, 1.0, upsamp, dims)
    xid = np.argmax(cmap, axis=1)
    iun = nclust * np.ones(dims, 'int')
    iclust = np.unravel_index(xid, iun)
    iclust = np.array(iclust)
    ipick = np.zeros((dims, N, M0.shape[1]))
    for j in range(dims):
        ipick[j] = (iclust[j][:, np.newaxis] + M0[j]) % nclust
    xid = np.ravel_multi_index(ipick.astype('int'), iun)
    iN = np.tile(np.arange(N)[:, np.newaxis], (1, M0.shape[1]))
    C0 = cmap[iN.astype('int'), xid]
    mu = np.mean(C0, axis=1)
    C0 = C0 - mu[:, np.newaxis]
    upC = C0 @ Km.T
    xid = np.argmax(upC, axis=1)
    cmax = np.amax(upC, axis=1) + mu
    dxs = M1[:, xid]
    xs = (iclust + dxs)/nclust
    xs = xs%1.
    xs = xs.T
    return xs, cmax


def dwrap(kx,nc):
    '''compute a wrapped distance'''
    q1 = np.mod(kx, nc)
    q2 = np.minimum(q1, nc-q1)
    return q2

def my_mesh(dims, xs):
    if dims==1:
        mesh = np.reshape(xs, (1,-1))
    else:
        mesh = my_mesh(dims-1, xs)
        m0 = xs[:, np.newaxis] * np.ones(mesh.shape[1])
        m0 = np.reshape(m0, (1, -1))
        mesh = np.tile(mesh, (1, len(xs)))
        mesh = np.concatenate((mesh, m0), axis = 0)
    return mesh

def upsampled_kernel(nclust, sig, upsamp, dims):
    # assume the input is 5 by 5 by 5 by 5.... vectorized

    r = int((nclust-1)/2)
    xs = np.linspace(-r,r,nclust)
    xn = np.linspace(-r,r,(nclust-1)*upsamp+1)
    M0 = my_mesh(dims, xs)
    M1 = my_mesh(dims, xn)

    d1 = np.zeros((M1.shape[1], M0.shape[1]))
    d0 = np.zeros((M0.shape[1], M0.shape[1]))
    for j in range(M0.shape[0]):
        d1 += (M1[j,:][:,np.newaxis] - M0[j,:])**2
        d0 += (M0[j,:][:,np.newaxis] - M0[j,:])**2
    K0 = np.exp(-d0/sig**2)
    K1 = np.exp(-d1/sig**2)
    Km = K1 @ np.linalg.inv(K0 + .01 * np.eye(K0.shape[0]))

    return Km, M1, M0

class Rastermap:
    """Rastermap embedding algorithm
    Rastermap takes the nPCs (400 default) of the data, and embeds them into
    n_X clusters. It returns upsampled cluster identities (n_X*upsamp).
    Optionally, a 1D embeddding is also computed across the second dimension (n_Y>0),
    smoothed over, and the PCA recomputed before fitting Rastermap.
    data X: n_samples x n_features
    Parameters
    -----------
    n_components : int, optional (default: 2)
        dimension of the embedding space
    alpha: float, optional (default: 1.0)
        exponent of the power law enforced on component n as: 1/(K+n)^alpha
    K:    float, optional (default: 1.0)
        additive offset of the power law enforced on component n as: 1/(K+n)^alpha
    n_X : int, optional (default: -1, i.e. auto)
        number of Fourier components used. If -1, 2/3 of the Fourier modes are used for each dimension.
    n_Y : int, optional (default: 0, i.e. not used)
        number of Fourier components in Y: will be used to smooth data for better PCs
    nPC : int, optional (default: 400)
        number of PCs to use during optimization
    init : initialization of algorithm (default: 'pca')
        can use 'pca', 'random', or a matrix n_samples x n_components
    verbose: bool (default: True)
        whether to output progress during optimization
    Returns
    ----------
    embedding : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
    u,sv,v : singular value decomposition of data S, potentially with smoothing
    isort1 : sorting along first dimension of matrix
    isort2 : sorting along second dimension of matrix (if n_Y > 0)
    cmap: correlation of each item with all locations in the embedding map (before upsampling)
    A:    PC coefficients of each Fourier mode
    """
    def __init__(self, n_components=2, n_X = -1, n_Y = 0,
                 nPC = 200, sig_Y=1.0, init='pca', alpha = 1., K = 1.,
                 mode = 'basic', verbose = True, annealing = True, constraints = 2):

        self.n_components = n_components
        self.alpha = alpha
        self.K     = K
        self.nPC = nPC
        self.sig_Y = sig_Y
        self.init = init
        self.mode = mode
        self.n_Y = n_Y
        self.constraints = constraints
        if constraints<2:
            self.annealing = 0
            self.init      = 'random'
        else:
            self.annealing = annealing
        if n_X>0:
            self.n_X = n_X
        else:
            self.n_X  = min(100, np.ceil(1600**(1/n_components)).astype('int'))

        self.verbose = verbose

    def fit(self, X, u=None, sv=None, v=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : Ignored
        """
        if self.mode is 'parallel':
            Xall = X.copy()
            X = np.reshape(Xall.copy(), (-1, Xall.shape[-1]))
        #X -= X.mean(axis=-1)[:,np.newaxis]
        if (u is None) or (sv is None) or (v is None):
            # compute svd and keep iPC's of data
            nmin = min([X.shape[0],X.shape[1]])
            nmin = np.minimum(nmin-1, self.nPC)
            u,sv,v = svdecon(np.float64(X), k=nmin)
            #u, sv, v = np.float32(u), np.float32(sv), np.float32(v)
        self.nPC = sv.size

        # first smooth in Y (if n_Y > 0)
        # this will be a 1-D fit
        isort2 = []
        if self.n_Y > 0:
            vsort = np.argsort(v[:,0])[:,np.newaxis]
            isort2, iclustup = self._map(v * sv, 1, self.n_Y, vsort)
            #X = gaussian_filter1d(X[:, isort2], self.sig_Y, axis=1)
            #u,sv,v = svdecon(np.float64(X), k=nmin)

        self.u = u
        self.sv = sv
        self.v = v
        if self.mode is 'parallel':
            NN = Xall.shape[1]
            X = np.zeros((2, NN, u.shape[1]), 'float64')
            for j in range(2):
                Xall[j] -= Xall[j].mean(axis=-1)[:, np.newaxis]
                X[j] = Xall[j] @ self.v
        else:
            NN = X.shape[0]
            X = X @ self.v

        if self.init == 'pca':
            u = u * np.sign(skew(u, axis=0))
            init_sort = np.argsort(u[:NN, :self.n_components], axis=0)
            #init_sort = u[:NN,:self.n_components]
            if False:
                ix = init_sort > 0
                iy = init_sort < 0
                init_sort[ix] = init_sort[ix] - 100.
                init_sort[iy] = init_sort[iy] + 100.
        elif self.init == 'random':
            init_sort = np.random.permutation(NN)[:,np.newaxis]
            for j in range(1,self.n_components):
                init_sort = np.concatenate((init_sort, np.random.permutation(NN)[:,np.newaxis]), axis=-1)
        else:
            init_sort = self.init
        if self.n_components==1 and init_sort.ndim==1:
            init_sort = init_sort[:,np.newaxis]

        # now sort in X
        isort1, iclustup = self._map(X, self.n_components, self.n_X, init_sort)
        self.isort2 = isort2
        self.isort1 = isort1
        self.embedding = iclustup
        return self

    def fit_transform(self, X, u=None, sv=None, v=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features). X contains a sample per row.
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
                AtS = self.A.T @ self.S
                vnorm   = np.sum(self.S * (self.A @ AtS), axis=0)[np.newaxis,:]
                cv      = X @ AtS
                cmap    = np.maximum(0., cv)**2 / vnorm
                iclustup, cmax = upsample(np.sqrt(cmap), dims, nclust, 10)
            else:
                print('ERROR: new points do not have as many features as original data')
        else:
            print('ERROR: need to fit model first before you can embed new points')
        if iclustup.ndim > 1:
            iclustup = iclustup.T
        else:
            iclustup = iclustup.flatten()
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
        if self.mode is 'parallel':
            Xall = X
            X = Xall[1]

        X -= X.mean(axis=0)
        NN,nPC = X.shape
        # initialize 1D clusters as nodes of 1st PC
        xid = np.zeros(NN)
        for j in range(dims):
            iclust = np.floor(nclust * u[:,j].astype(np.float64)/NN)
            xid = nclust * xid + iclust
        xid = xid.astype('int').flatten()

        if self.constraints==0:
            nfreqs = nclust
        elif self.constraints==1:
            nfreqs = np.ceil(1/2 * nclust)
            nfreqs = int(2 * np.floor(nfreqs/2)+1)
        else:
            nfreqs = np.ceil(2/3 * nclust)
            nfreqs = int(2 * np.floor(nfreqs/2)+1)

        if dims>1:
            SALL, fxx = create_ND_basis(dims, nclust, nfreqs, True)
        else:
            SALL, fxx = create_ND_basis(dims, nclust, nfreqs, False)
        SALL = SALL[1:, :]
        fxx = fxx[1:]
        tic = time.time()

        if self.annealing:
            nskip = int(2 * max(1., nfreqs/100))
            ncomps_anneal = (np.arange(3, nfreqs, nskip)**dims).astype('int')  - 1
            ncomps_anneal = np.tile(ncomps_anneal, (2,1)).T.flatten()
            ncomps_anneal = np.concatenate((ncomps_anneal[:10], ncomps_anneal[2:10], ncomps_anneal[4:], SALL.shape[0]*np.ones(20)), axis=0).astype('int')
        else:
            ncomps_anneal = SALL.shape[0]*np.ones(50).astype('int')
        #ncomps_anneal = np.concatenate((ncomps_anneal[1:5], ncomps_anneal[1:10], ncomps_anneal[3:], SALL.shape[0]*np.ones(10)), axis=0).astype('int')
        #ncomps_anneal = np.concatenate((ncomps_anneal[1:], SALL.shape[0]*np.ones(20)), axis=0).astype('int')
        #ncomps_anneal = np.concatenate((np.linspace(4,SALL.shape[0],30), SALL.shape[0]*np.ones(20)), axis=0).astype('int')

        # vscale = (self.K + fxx[:nc])**2

        vscale = self.K + np.arange(len(fxx))

        xnorm = (X**2).sum(axis=1)[:,np.newaxis]
        E = np.zeros(len(ncomps_anneal)+1)
        if self.verbose:
            print('time; iteration;  explained PC variance')
        if self.mode is 'parallel':
            cmapx = np.zeros((2, NN, nclust**dims), 'float32')

        for t,nc in enumerate(ncomps_anneal[:2]):
            # get basis functions
            S = SALL[:nc, :]
            S0 = S[:, xid]
            A  = S0 @ X
            nA      = np.sum(A**2, axis=1)**.5
            nA      *= vscale[:nc]**(self.alpha/2)
            if self.constraints<2:
                nA = np.ones(nA.shape)
            A        /= nA[:, np.newaxis]
            eweights = (S0.T / nA) @ S
            AtS     = A.T @ S
            vnorm   = np.sum(S * (A @ AtS), axis=0)[np.newaxis,:]
            if self.mode=='parallel':
                X = Xall[t%2]
            cv      = X @ AtS
            vnorm   = vnorm + xnorm  * eweights**2 - 2*eweights * cv
            cv      = cv - xnorm * eweights
            cmap    = np.maximum(0., cv)**2 / vnorm
            cmax    = np.amax(cmap, axis=1)
            xid     = np.argmax(cmap, axis=1)
            E[t]    = np.nanmean(cmax)/np.nanmean(xnorm)
            if self.mode is 'parallel':
                cmapx[t%2] = cmap
            if t%10==0:
                if self.verbose:
                    print('%2.2fs    %2.0d        %4.4f'%(time.time()-tic, t, E[t]))
        if self.verbose:
            print('%2.2fs   final      %4.4f'%(time.time()-tic, E[t]))
        if self.mode is 'parallel':
            iclustup1, cmax = upsample(np.sqrt(cmapx[0]), dims, nclust, 10)
            iclustup2, cmax = upsample(np.sqrt(cmapx[1]), dims, nclust, 10)
            iclustup = np.concatenate((iclustup1[np.newaxis, :, :], iclustup2[np.newaxis, :, :]), axis=0)
            isort = np.argsort(iclustup2[:,0])
            self.cmap = cmapx
        else:
            iclustup, cmax = upsample(np.sqrt(cmap), dims, nclust, 10)
            isort = np.argsort(iclustup[:,0])
            self.cmap = cmap
        E[t+1] = np.nanmean(cmax**2)/np.nanmean(xnorm)
        if self.verbose:
            print('%2.2fs upsampled    %4.4f'%(time.time()-tic, E[t+1]))
        self.E = E
        self.S = S
        self.A = A
        self.xid = xid
        return isort, iclustup

def main(X,ops=None,u=None,sv=None,v=None):
    if u is None:
        nmin = min([X.shape[0],X.shape[1]])
        nmin = np.minimum(nmin-1, ops['nPC'])
        sv,u = eigsh(X @ X.T, k=nmin)
        sv = sv**0.5
        v = X.T @ u
    isort2,iclust2 = _map(X.T,ops,v,sv)
    Xm = X - X.mean(axis=1)[:,np.newaxis]
    Xm = gaussian_filter1d(Xm,3,axis=1)
    isort1,iclust1 = _map(Xm,ops,u,sv)
    return isort1,isort2
