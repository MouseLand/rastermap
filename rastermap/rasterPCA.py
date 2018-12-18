from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore, skew
import math
import numpy as np
import time

def swap_lines(CC0, di):
    npix = CC0.shape[0]
    irange = np.arange(npix-1)
    ri = np.arange(1,npix-di-1)
    rj = ri+di
    m0 = np.logical_and(ri[:,np.newaxis]-2<irange, rj[:,np.newaxis]+1>irange)
    m0 = np.logical_not(m0).astype('float32')
    Cost = CC0[ri-1,rj+1]- CC0[ri-1,ri] - CC0[rj,rj+1]
    Cost1 = Cost[:,np.newaxis] + CC0[np.ix_(ri,irange)] + CC0[np.ix_(rj,irange+1)] - CC0[irange, irange+1]
    Cost2 = Cost[:,np.newaxis] + CC0[np.ix_(rj,irange)] + CC0[np.ix_(ri,irange+1)] - CC0[irange, irange+1]
    Cost3 = CC0[ri-1,rj] + CC0[rj+1,ri] - CC0[ri-1,ri] - CC0[rj,rj+1]
    #Cost2 = Cost2* 0
    Cost3 = Cost3* 0
    Cost12  = np.maximum(Cost1, Cost2)
    Cost123 = np.maximum(Cost12, Cost3[:, np.newaxis]) * m0
    Cmax = np.amax(Cost123)
    imax = np.argmax(Cost123)
    x,y  = np.unravel_index(imax, Cost123.shape)
    if Cost3[x]>Cost12[x,y]:
        flip = 2
    else:
        flip = Cost1[x,y] < Cost2[x,y]
    return Cmax, ri[x], irange[y], flip

def bin(X0, dt):
    NN, NT = X0.shape
    NN = int(dt * np.floor(NN/dt))
    X0 = X0[:NN, :]
    X0 = np.reshape(X0, (-1, dt, NT)).mean(axis=1)
    return X0

def resort_X(X0, niter=500):
    X = X0.copy()
    npix, nd = X.shape
    X = np.vstack((X[0,:], X, X[0,:]))
    npix, nd = X.shape
    CC = np.corrcoef(X)
    CC[0, :] = 0
    CC[-1, :] = 0
    CC[:, 0] = 0
    CC[:, -1] = 0
    xid = np.arange(npix)
    for k in range(niter):
        flag = 0
        for di in np.arange(1, npix-2):
            Cmax, xstart, xinsrt, flip  = swap_lines(CC, di)
            if Cmax>0:
                # do the swap if it improves cost
                iall = np.hstack((np.arange(xstart), np.arange(xstart+di+1,npix)))
                if xstart+di+1>=npix:
                    print(xstart, di)
                ifind = int(np.nonzero(iall==xinsrt)[0])
                if ifind+1>=len(iall):
                    print(xstart, di, ifind, len(iall))
                if flip==2:
                    isort = np.hstack((np.arange(xstart), np.arange(xstart, xstart+di+1)[::-1], np.arange(xstart+di+1,npix)))
                elif flip==1:
                    isort = np.hstack((iall[:ifind+1], np.arange(xstart, xstart+di+1)[::-1], iall[ifind+1:]))
                else:
                    isort = np.hstack((iall[:ifind+1], np.arange(xstart, xstart+di+1), iall[ifind+1:]))
                CC = CC[np.ix_(isort, isort)]
                X = X[isort, :]
                #xid1 = np.zeros(npix, 'int32')
                #xid1[isort] = xid
                xid = xid[isort]
                flag = 1
                break
        if flag==0:
            break
    print(k, flag)
    X = X[1:-1, :]
    xid = xid[1:-1] - 1
    return X, xid

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
    #flag = False
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

class rasterPCA:
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
            nmin = min([X.shape[0], X.shape[1]])
            nmin = np.minimum(nmin-1, self.nPC)
            u,sv,v = svdecon(np.float64(X), k=nmin)
            u = u * sv
        NN, self.nPC = u.shape
        self.u = u
        # now sort in X
        U = self._map(u.copy(), self.n_components, self.n_X, u.copy())
        return self

    def _map(self, X, dims, nclust, uPCA):
        #X -= X.mean(axis=0)

        if self.constraints==0:
            nfreqs = nclust
        elif self.constraints==1:
            nfreqs = np.ceil(3/4 * nclust)
            nfreqs = int(2 * np.floor(nfreqs/2)+1)
        else:
            nfreqs = np.ceil(2/3 * nclust)
            nfreqs = int(2 * np.floor(nfreqs/2)+1)

        if dims>1:
            SALL, fxx = create_ND_basis(dims, nclust, nfreqs, True)
        else:
            SALL, fxx = create_ND_basis(dims, nclust, nfreqs, False)
        #SALL = SALL[1:, :]
        #fxx = fxx[1:]
        tic = time.time()

        if self.annealing:
            nskip = int(2 * max(1., nfreqs/100))
            ncomps_anneal = (np.arange(3, nfreqs, nskip)**dims).astype('int')  - 1
            ncomps_anneal = np.tile(ncomps_anneal, (2,1)).T.flatten()
            ncomps_anneal = np.concatenate((ncomps_anneal[:10], ncomps_anneal[2:10], ncomps_anneal[4:], SALL.shape[0]*np.ones(20)), axis=0).astype('int')
        else:
            ncomps_anneal = SALL.shape[0]*np.ones(50).astype('int')

        nbasis,npix = SALL.shape
        print(ncomps_anneal.shape)

        if self.constraints==2:
            self.vscale = 1/(self.K + np.arange(len(fxx)))**(self.alpha/2)

        xnorm = (X**2).sum(axis=1)[:,np.newaxis]
        E = np.zeros(len(ncomps_anneal)+1)
        if self.verbose:
            print('time; iteration;  explained PC variance')

        u = uPCA[:, :nclust]
        u /= np.sum(u**2, axis=0)**.5
        for t,nc in enumerate(ncomps_anneal):
            S = SALL[:nc, :]
            A = S @ (u.T @ X)
            V = S.T @ A

            u = np.linalg.solve(V @ V.T + np.eye(nclust), V @ X.T).T
            E[t] = 1 - np.sum((X - u @ V)**2)/xnorm.sum()

            u /= np.sum(u**2, axis=0)**.5

            if t%10==0:
                if self.verbose:
                    print('%2.2fs    %2.0d        %4.4f'%(time.time()-tic, t, E[t]))

        if self.verbose:
            print('%2.2fs   final      %4.4f'%(time.time()-tic, E[t]))
        self.E = E
        self.S = S
        self.A = A
        self.u = u
        return u

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
