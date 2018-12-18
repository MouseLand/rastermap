from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore, skew
import math
import numpy as np
import time


def bin(X0, dt):
    NN, NT = X0.shape
    NN = int(dt * np.floor(NN/dt))
    X0 = X0[:NN, :]
    X0 = np.reshape(X0, (-1, dt, NT)).mean(axis=1)
    return X0

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

def create_ND_basis(dims, nclust):
    # recursively call this function until we fill out S
    #flag = False
    if dims==1:
        xs = np.arange(nclust)/nclust
    elif dims==2:
        xs, ys = np.meshgrid(np.arange(nclust)/nclust, np.arange(nclust)/nclust)
        xs, ys = xs.flatten(), ys.flatten()
        xs = np.hstack((xs[:, np.newaxis], ys[:, np.newaxis]))
    ds = distances(xs, xs)
    return ds

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
    alpha : float, optional (default: 1.0)
        exponent of the power law enforced on component n as: 1/(K+n)^alpha
    K :  float, optional (default: 1.0)
        additive offset of the power law enforced on component n as: 1/(K+n)^alpha
    n_X :  int, optional (default: 40)
        size of the grid on which the Fourier modes are rasterized
    n_Y : int, optional (default: 0, i.e. not used)
        number of Fourier components in Y: will be used to smooth data for better PCs
    nPC : int, optional (default: 400)
        number of PCs to use during optimization
    init : initialization of algorithm (default: 'pca')
        can use 'pca', 'random', or a matrix n_samples x n_components
    verbose: bool (default: True)
        whether to output progress during optimization
    """
    def __init__(self, n_components=2, n_X = 40,
                 nPC = 200, init='pca', alpha = 1., K = 1.,
                 mode = 'basic', verbose = True, annealing = True, constraints = 2):

        self.n_components = n_components
        self.alpha = alpha
        self.K     = K
        self.nPC = nPC
        self.init = init
        self.mode = mode
        self.constraints = constraints
        #if constraints<2:
            #self.annealing = 0
            #self.init      = 'random'
        #else:
        self.annealing = annealing
        self.n_X  = int(n_X)
        self.verbose = verbose

    def fit(self, X=None, u=None):
        """Fit X into an embedded space.
        Inputs
        ----------
        X : array, shape (n_samples, n_features)
        u,s,v : svd decomposition of X (optional)

        Assigns
        ----------
        embedding : array-like, shape (n_samples, n_components)
            Stores the embedding vectors.
        u,sv,v : singular value decomposition of data S, potentially with smoothing
        isort1 : sorting along first dimension of matrix
        isort2 : sorting along second dimension of matrix (if n_Y > 0)
        cmap: correlation of each item with all locations in the embedding map (before upsampling)
        A:    PC coefficients of each Fourier mode

        """
        X = X.copy()
        if self.mode is 'parallel':
            Xall = X.copy()
            X = np.reshape(Xall.copy(), (-1, Xall.shape[-1]))
        #X -= X.mean(axis=-1)[:,np.newaxis]
        if ((u is None)):
            # compute svd and keep iPC's of data
            nmin = min([X.shape[0], X.shape[1]])
            nmin = np.minimum(nmin-1, self.nPC)
            u,sv,v = svdecon(np.float64(X), k=nmin)
            u = u * sv

        NN, self.nPC = u.shape
        if self.constraints==3:
            plaw = 1/(1+np.arange(1000))**(self.alpha/2)
            self.vscale = np.sum(u**2,axis=0)**.5
            tail = self.vscale[-1] * plaw[u.shape[1]:]/plaw[u.shape[1]]
            self.vscale = np.hstack((self.vscale, tail))
        # first smooth in Y (if n_Y > 0)
        self.u = u

        if self.mode is 'parallel':
            NN = Xall.shape[1]
            X = np.zeros((2, NN, u.shape[1]), 'float64')
            for j in range(2):
                Xall[j] -= Xall[j].mean(axis=-1)[:, np.newaxis]
                X[j] = Xall[j] @ self.v

        if self.init == 'pca':
            usort = u * np.sign(skew(u, axis=0))
            init_sort = np.argsort(usort[:NN, :self.n_components], axis=0)
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
        isort1, iclustup = self._map(u.copy(), self.n_components, self.n_X, init_sort)
        self.isort = isort1
        self.embedding = iclustup
        return self

    def fit_transform(self, X, u=None, sv=None, v=None):
        """Fit X into an embedded space and return that transformed
        output.
        Inputs
        ----------
        X : array, shape (n_samples, n_features). X contains a sample per row.

        Returns
        -------
        embedding : array, shape (n_samples, n_components)
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

    def _map(self, X, dims, nclust, u):
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
            nfreqs = np.ceil(1/4 * nclust)
            nfreqs = int(2 * np.floor(nfreqs/2)+1)
        else:
            nfreqs = np.ceil(2/3 * nclust)
            nfreqs = int(2 * np.floor(nfreqs/2)+1)

        ds = create_ND_basis(dims, nclust)
        npix = ds.shape[1]

        X -= X.mean(axis=0)
        tic = time.time()

        full_pass = np.linspace(5., 1, 30)/nclust
        phase1 = full_pass[:10]
        phase2 = full_pass[:10]
        phaseX = full_pass[-1] * np.ones(10)
        sig = np.hstack((phase1, phase2, full_pass[3:], phaseX))


        xnorm = (X**2).sum(axis=1)[:,np.newaxis]
        E = np.zeros(len(sig)+1)
        if self.verbose:
            print('time; iteration;  explained PC variance')

        lam = np.ones(NN)

        for t,nc in enumerate(sig):
            # get basis functions
            StS = np.exp(-ds**2/(sig[t]**2))
            #StS = np.exp(-ds/sig[t])

            X0 = np.zeros((npix, nPC))
            for j in range(npix):
                ix = xid==j
                if np.sum(ix):
                    lam[ix] /= np.sum(lam[ix]**2)**.5
                    X0[j, :] = lam[ix] @ X[ix, :]
            AtS = (StS @ X0).T
            eweights = StS[xid, :] * lam[:, np.newaxis]
            cv      = X @ AtS
            vnorm   = np.sum(AtS**2, axis=0)[np.newaxis,:]
            vnorm   = vnorm + xnorm  * eweights**2 - 2*eweights * cv
            cv      = cv - xnorm * eweights
            cmap    = np.maximum(0.0001, cv) **2 / vnorm
            cmax    = np.amax(cmap, axis=1)
            xid     = np.argmax(cmap, axis=1)

            #lam = np.sqrt(cmax /vnorm[0,xid])
            lam    = np.sqrt(cmax / vnorm[np.arange(NN), xid])

            E[t]    = np.nanmean(cmax)/np.nanmean(xnorm)
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
        self.StS = StS
        self.AtS = AtS
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
