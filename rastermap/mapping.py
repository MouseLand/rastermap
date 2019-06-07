from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore, skew
from sklearn.decomposition import PCA
import math
import numpy as np
import time
import multiprocessing
from multiprocessing import Pool
from numba import jit


def upsample_grad2(AtS, X, xid, dims, nX):
    ys, xs = np.meshgrid(np.arange(nX), np.arange(nX))
    xygrid = np.vstack((xs.flatten(), ys.flatten()))

    NN = xs.shape[0]
    eta = .1
    sig2 = 1.
    #

    dy = np.zeros((2, NN))
    y = xygrid[xid, :]
    ds = np.zeros((2, y.shape[0], xygrid.shape[0]))
    for j in range(2):
        ds[j] = y[:, j][:, np.newaxis] - xygrid[:,j]

    W = exp(-np.sum(ds**2, axis=0)/sig2)
    ypred = W @ AtS
    lam = np.sum(ypred * X, axis=1)/np.sum(ypred**2, axis=1)
    err = lam[:, np.newaxis] * ypred - X
    err = err * lam[:, np.newaxis]

    for j in range(2):
        dy[j] = np.sum(err * W * ds[j], axis=1)
    y = y - eta[j] * err

    return y

def upsample_grad(CC, dims, nX):
    CC /= np.amax(CC, axis=1)[:, np.newaxis]
    xid = np.argmax(CC, axis=1)

    ys, xs = np.meshgrid(np.arange(nX), np.arange(nX))
    y0 = np.vstack((xs.flatten(), ys.flatten()))

    eta = .1
    y = optimize_neurons(CC, y0[:,xid], y0, eta)
    return y

def get_border(dims, nclust):
    if dims==2:
        mask = np.zeros((nclust, nclust), 'bool')
        mask[:1,:] = True
        mask[:, :1] = True
        mask[-1:,:] = True
        mask[:,-1:] = True
    mask = mask.flatten()
    return mask

def shrink_to_center(cmap, dims, nclust, upsamp, fact0):
    #fact0 = .8 #1 - 1./10
    if dims==2:
        xs, cmax    = upsample(cmap, dims, nclust, upsamp)
        xs          = xs * nclust
        xhalf       = (nclust-1)/2

        xs = xs + (xhalf - np.mean(xs,axis=0))

        dx = xs - xhalf
        fact = fact0 * (xhalf/2) / np.mean(np.abs(dx), axis=0)

        #print(xhalf, fact,  np.amax(xs, axis=0), np.amin(xs, axis=0))
        Xs          = xhalf + fact * (xs-xhalf)
        xs          = np.round(Xs).astype('int')
        xs = np.maximum(0, xs)
        xs = np.minimum(nclust-1, xs)
        xid = np.ravel_multi_index((xs[:,0], xs[:,1]), (nclust, nclust))
    return xid

def add_noise(xid, nclust, dims):
    if dims==2:
        ys, xs = np.unravel_index(xid, (nclust,nclust))
        ys = ys + f[t] * nclust * (np.random.rand(ys.size)-.5)
        xs = xs + f[t] * nclust * (np.random.rand(ys.size)-.5)
        xs = xs.astype('int')
        ys = ys.astype('int')
        xs = np.maximum(0, xs)
        ys = np.maximum(0, ys)
        xs = np.minimum(nclust-1, xs)
        ys = np.minimum(nclust-1, ys)
        xid = ys + nclust * xs
    return  xid

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

@jit(nopython=True, nogil=True)
def bin2(X0, dt, axis=0):
    NN, NT = X0.shape
    if axis==0:
        NN = int(dt * np.floor(NN/dt))
        X = np.zeros((int(NN/dt), NT), np.float32)
        for j in range(NN/dt):
            X[j, :] = np.sum(X0[j*dt:(j+1)*dt, :], axis=0)
    else:
        NT = int(dt * np.floor(NT/dt))
        X = np.zeros((NN, int(NT/dt)), np.float32)
        for j in range(NT/dt):
            X[:, j] = np.sum(X0[:, j*dt:(j+1)*dt], axis=1)
    X = X / dt
    return X

def bin(X0, dt, axis=0):
    NN, NT = X0.shape
    if axis==0:
        NN = int(dt * np.floor(NN/dt))
        X = np.reshape(X0[:NN, :], (-1, dt, NT)).mean(axis=1)
    else:
        NT = int(dt * np.floor(NT/dt))
        X = np.reshape(X0[:, :NT], (NN, -1, dt)).mean(axis=2)

    return X

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
        ds += (x[:,j][:,np.newaxis] - y[:,j])**2
    ds = np.sqrt(ds)
    return ds

def create_ND_basis(dims, nclust, K, flag=True):
    # recursively call this function until we fill out S
    flag = True
    if dims==1:
        xs = np.arange(0,nclust)
        S = np.ones((K, nclust), 'float64')
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
    COV = np.float64(COV)
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
    if NN>NT:
        Sv *= NT**.5
    else:
        Sv *= NN**.5

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
    xs = (iclust + dxs[1::-1, :])/nclust
    #xs = xs%1.
    xs = xs.T
    return xs, cmax

@jit(nopython=True, nogil=True)
def my_matmul(x,y):
    z = x @ y
    return z

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

def expand_dij(dist, ind, dij):
    ncores = 10
    #nbatches = ncores
    NN = dist.shape[0]
    #nbatch = int(np.ceil(NN/ncores))
    nbatch = 1000
    nbatches = int(np.ceil(NN/nbatch))
    inputs = []
    indx = []
    for j in range(nbatches):
        ilist = np.arange(j*nbatch, min(NN, (j+1)*nbatch))
        indx.append(ilist)
        inputs.append((dist[indx[j], :], ind[indx[j], :], dij))

    with Pool(ncores) as p:
        y = p.map(expand_worker, inputs)

    dij0 = np.zeros((dij.shape[0], NN))
    for j in range(nbatches):
        dij0[:, indx[j]] = y[j]

    # dij = np.amin(np.expand_dims(dist[ :10000, :], 0)  + dij[:, ind[:10000, :]], axis=-1)
    return dij0

def expand_worker(inputs):
    dist, ind, dij = inputs
    dij = np.amin(dist[np.newaxis, :, :]  + dij[:, ind[:,:]], axis=-1)
    return dij


@jit(nopython=True, nogil=True)
def my_max(a, b):
    abmax = np.maximum(a, b)
    return abmax

def scaled_kmeans(X, k=32):
    # reduces data to k clusters by scaled k-means
    NN, nf = X.shape
    # initialize with cluster means randomly
    xmeans = X[np.random.permutation(NN)[:k], :]
    niter = 20
    lam = np.ones(NN)
    xid = np.linspace(0, k-1, NN).astype('int')
    for i in range(niter):
        xmeans /= (1e-10 + np.sum(xmeans**2, axis=1))[:, np.newaxis]**.5
        cv = X @ xmeans.T
        cv = my_max(0., cv)
        xid = np.argmax(cv, axis = 1)
        lam = cv[np.arange(NN), xid]
        for j in range(k):
            ix = xid==j
            if np.sum(ix):
                xmeans[j, :] = lam[ix] @ X[ix, :]
    return xmeans, xid

def nearest_neigh(X0, X1, k= 32):
    upow = np.sum(X0**2, axis=1)
    upow2 = np.sum(X1**2, axis=1)

    utu  = upow + upow2[:, np.newaxis] - 2 * X1 @ X0.T
    ind = np.argpartition(utu, k, axis=1)
    ind = ind[:, :k]
    dist = utu[np.tile(np.arange(X1.shape[0])[:, np.newaxis], (1, k)).flatten(), ind.flatten()]

    dist = np.reshape(dist, (X1.shape[0], -1))
    #isort = np.argsort(dist, axis=1).astype('int')
    #ind = ind[np.tile(np.arange(X1.shape[0])[:, np.newaxis], (1, k)).flatten(), isort.flatten()]
    #ind = np.reshape(ind, (X1.shape[0], -1))

    dist = np.maximum(0, dist)**.5
    return dist, ind

def dbisomap(X, k=32, nISO = 256, d = 2):
    NN = X.shape[0]
    NMAX = min(20000, NN)
    n0   = min(128, NN)

    t0 = time.time()

    # get the landmarks from k-means
    xmeans, xid = scaled_kmeans(X, k=n0)
    xmeans = zscore(xmeans, axis=1)/xmeans.shape[1]**.5
    #xmeans = X[np.linspace(0, NN-1, n0).astype('int32'), :]
    print('time %2.2f. found landmarks'%(time.time()-t0))

    X0    = np.concatenate((xmeans, X), axis=0)
    iland = np.arange(n0)
    itree = n0 + np.linspace(0, NN-1, NMAX).astype('int32')
    itree = np.hstack((iland, itree))

    dist, ind = nearest_neigh(X0[itree, :], X0, k)
    #tree        = cKDTree(X0[itree, :], leafsize=40)
    #dist, ind   = tree.query(X0, k=k, eps= 2, n_jobs=-1)
    print('time %2.2f. found nearest neighbors'%(time.time()-t0))

    pho     = np.amax(dist, axis=1)
    pho     = np.maximum(pho, np.median(pho)/10)
    pho     = 1/pho
    dist    = dist * (pho[:, np.newaxis] * pho[ind])**(d/2)
    ix      = np.tile(np.arange(NMAX+n0)[:, np.newaxis], (1, dist.shape[1]))
    D       = csr_matrix((dist[itree].flatten(), (ix.flatten(), ind[itree].flatten())), shape=(NMAX+n0, NMAX+n0))
    D       = lil_matrix(D)
    Dt      = D.T
    D[Dt > D]    = Dt[Dt > D]
    print('time %2.2f. contructed sparse matrix'%(time.time()-t0))

    dij          = dijkstra_pool(D, iland)
    print('time %2.2f. computed dijkstra'%(time.time()-t0))

    dij[dij>100] = 100
    # now expand dij to all the other points
    dij = expand_dij(dist[n0:, :], ind[n0:, :], dij)
    print('time %2.2f. upsampled'%(time.time()-t0))

    H = -(dij**2).T
    H -= np.mean(H, axis=1)[:, np.newaxis]
    H -= np.mean(H, axis=0)
    v = PCA(n_components=min(n0, nISO)).fit(H).components_
    V = H @ v.T
    S = np.sum(V**2, axis=0)**.5
    V /= S
    print('time %2.2f. finished'%(time.time()-t0))

    return S, V#, ind[n0:, :]-n0, itree[n0:] -n0

def dijkstra_pool(D, irange):
    inputs = []
    nbatches = 10
    for j in range(nbatches):
        inputs.append((D, irange[j::nbatches]))

    num_cores = nbatches # multiprocessing.cpu_count()
    with Pool(num_cores) as p:
        y = p.map(dijkstra_worker, inputs)

    dij = np.zeros((len(irange), D.shape[0]))
    for j in range(len(y)):
        dij[j:len(irange):nbatches, :] = y[j]

    return dij

def dijkstra_worker(inputs):
    D, ikp = inputs
    dij = dijkstra(D, indices=ikp, directed = False)
    return dij

@jit(nopython=True, nogil=True)
def assign_neurons2(vnorm, cv):
    cmap    = np.maximum(0., cv)**2 / vnorm
    return cmap

@jit(nopython=True, nogil=True)
def assign_neurons(vnorm, xnorm, eweights, cv):
    vnorm   = vnorm + xnorm  * eweights**2 - 2*eweights * cv
    cv      = cv - xnorm * eweights
    cmap    = np.maximum(0., cv)**2 / vnorm
    return cmap, vnorm

def gradient_descent_neurons(inputs):
    CC, yinit, ynodes, eta = inputs
    flag = 1
    niter = 201 # 201
    alpha = 1.
    y = yinit
    x = 1.
    sig = 1.
    eta = np.linspace(eta, eta/10, niter)
    for j in range(niter):
        yy0 =  y[:, np.newaxis] - ynodes
        if flag:
            K = np.exp(-np.sum(yy0**2, axis=0)/(2*sig**2))
        else:
            yyi = 1 + np.sum(yy0**2, axis=0)
            K = 1/yyi**alpha
        x = np.sum(K*CC)/np.sum(K**2)
        err = (x*K - CC)
        if flag:
            Kprime = - x * yy0 * K
        else:
            Kprime = - yy0 * alpha * 1/yyi**(alpha+1)
        dy = np.sum(Kprime *err, axis=-1)
        y = y - eta[j] * dy
    return y

def optimize_neurons(CC, y, ynodes, eta):
    inputs = []
    for j in range(CC.shape[0]):
        inputs.append((CC[j,:], y[:, j], ynodes, eta))

    num_cores = multiprocessing.cpu_count()
    with Pool(num_cores) as p:
        y = p.map(gradient_descent_neurons, inputs)
    #y = gradient_descent_neurons((CC, y, ynodes, eta))

    y = np.array(y).T
    return y

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

    def fit_transform(self, X, u=None):
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
        self.fit(X, u)
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

    def fit(self, X=None, u=None, s=None):
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
        t0 = time.time()
        if self.mode is 'parallel':
            #X = X.copy()
            d = int(X.shape[1]/5e4)

            #X -= np.expand_dims(X.mean(axis=1), 1)
            #X0 = X[0].T @ X[1]

            X0 = X[:, ::d, :]
            X0 -= np.expand_dims(X0.mean(axis=1), 1)
            X0 = X0[0].T @ X0[1]

            X0 = X0 + X0.T
            NN = X.shape[1]
        else:
            X = X -  np.mean(X, axis=0)
            d = max(1, int(X.shape[0]/5e4))
            X0 = X[::d, :]
            NN = X.shape[0]

        if ((u is None)):
            # compute svd and keep iPC's of data
            nmin = np.amin(X0.shape)
            nmin = np.minimum(nmin, self.nPC)
            print("nmin %d"%nmin)
            print(time.time() - t0)
            model = PCA(n_components=nmin).fit(X0)
            #v = svdecon(X0, k=nmin)[2].T
            print(time.time() - t0)
            v = model.components_
            #u = u * sv
            self.v = v
            #self.u = u
            self.nPC = nmin

        if self.mode is 'parallel':
            self.u = np.zeros((2, X.shape[1], self.nPC), np.float32)
            self.u[0] = X[0] @ v.T
            self.u[1] = X[1] @ v.T
            self.u -= np.expand_dims(self.u.mean(axis=1), 1)
        else:
            self.u = X @ v.T
        print(time.time() - t0)

        if self.constraints==3:
            plaw = 1/(1+np.arange(1000))**(self.alpha/2)
            self.vscale = np.sum(u**2,axis=0)**.5
            tail = self.vscale[-1] * plaw[u.shape[1]:]/plaw[u.shape[1]]
            self.vscale = np.hstack((self.vscale, tail))

        nclust = self.n_X
        if self.init == 'pca':
            if self.mode is 'parallel':
                usort = self.u[0] * np.sign(skew(self.u[0], axis=0))
            else:
                usort = self.u * np.sign(skew(self.u, axis=0))
                #self.u = zscore(self.u, axis=1)/self.u.shape[1]**.5
        elif self.init == 'dbisomap':
                self.u = zscore(self.u, axis=1)/self.u.shape[1]**.5
                S, self.u = dbisomap(self.u, k=32, nISO = nmin)
                self.u = self.u * S
                usort = self.u
                #S, usort = dbisomap(self.u, k = 32)
                usort = usort * np.sign(skew(usort, axis=0))
        elif self.init == 'random':
            init_sort = np.random.permutation(NN)[:,np.newaxis]
            for j in range(1,self.n_components):
                init_sort = np.concatenate((init_sort, np.random.permutation(NN)[:,np.newaxis]), axis=-1)
            xid = np.zeros(NN)
            for j in range(self.n_components):
                iclust = np.floor(nclust * init_sort[:,j].astype(np.float64)/NN)
                xid = nclust * xid + iclust
        elif self.init =='laplacian':
            Uz = zscore(self.u, axis=1)/self.u.shape[1]**.5
            CC = Uz @ Uz.T
            CCsort = np.sort(CC, axis=0)[::-1, :]
            CC[CC<CCsort[100, :]] = 0
            CC = (CC + CC.T)/2
            M = np.diag(np.sum(CC, axis=1)) - CC
            S, V = eigsh(M, k=3, which='SM')
            usort = V[:, 1:]
            usort = usort * np.sign(skew(usort, axis=0))
        elif self.init=='lle':
            Uz = zscore(self.u, axis=1)/self.u.shape[1]**.5
            CC = Uz @ Uz.T
            CCsort = np.sort(CC, axis=1)
            W = np.float64(CC > CCsort[:, -64][:, np.newaxis])
            W = W - np.diag(np.diag(W))
            W /= np.sum(W, axis=1)[:, np.newaxis]
            M = np.eye(W.shape[0]) - W
            S, V = eigsh(M.T @ M, k=3, which='SM')
            usort = V[:, 1:]
            usort = usort * np.sign(skew(usort, axis=0))
        else:
            init_sort = self.init
            xid = np.zeros(NN)
            for j in range(self.n_components):
                iclust = np.floor(nclust * init_sort[:,j].astype(np.float64)/NN)
                xid = nclust * xid + iclust

        if self.init=='pca' or self.init=='laplacian' or self.init=='lle' or self.init=='dbisomap':
            init_sort = np.argsort(usort[:, :self.n_components], axis=0)
            xid = np.zeros(NN)
            for j in range(self.n_components):
                iclust = np.floor(nclust * init_sort[:,j].astype(np.float64)/NN)
                xid = nclust * xid + iclust

        xid = xid.astype('int').flatten()

        #self.init_sort = usort

        if self.n_components==1 and init_sort.ndim==1:
            init_sort = init_sort[:,np.newaxis]

        # now sort in X
        print(time.time() - t0)
        isort1, iclustup = self._map(self.u, self.n_components, self.n_X, xid, s)
        self.isort = isort1
        self.embedding = iclustup
        return self

    def _map(self, X, dims, nclust, xid, SALL=None):
        if self.mode is 'parallel':
            Xall = X
            X = Xall[1]

        NN,nPC = X.shape
        # initialize 1D clusters as nodes of 1st PC

        if self.constraints==0:
            nfreqs = nclust
        elif self.constraints==1:
            nfreqs = np.ceil(1 * nclust)
            nfreqs = int(2 * np.floor(nfreqs/2)+1)
        else:
            nfreqs = nclust-1
            #nfreqs = np.ceil(1/2 * nclust)
            nfreqs = int(2 * np.floor(nfreqs/2)+1)

        if SALL is None:
            if dims>1:
                SALL, fxx = create_ND_basis(dims, nclust, nfreqs, True)
            else:
                SALL, fxx = create_ND_basis(dims, nclust, nfreqs, False)
            SALL = SALL[1:, :]
            fxx = fxx[1:]
        else:
            #SALL = SALL[:nfreqs**2-1, :]
            SALL = SALL[:nfreqs**2-1, :]
            fxx = np.arange(SALL.shape[0])
        print(SALL.shape)

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
            self.vscale = 1 + np.arange(len(fxx))
            self.vscale = 2 * np.ceil(self.vscale/2)
            self.vscale = 1/self.vscale**(self.alpha/2)
            print(self.alpha)

        xnorm = (X**2).sum(axis=1)[:,np.newaxis]
        E = np.zeros(len(ncomps_anneal)+1)
        if self.verbose:
            print('time; iteration;  explained PC variance')
        if self.mode is 'parallel':
            cmapx = np.zeros((2, NN, nclust**dims), 'float32')

        tic = time.time()

        lam = np.ones(NN)
        X0 = np.zeros((npix, nPC))
        for t,nc in enumerate(ncomps_anneal):
            # get basis functions
            S = SALL[:nc, :]
            for j in range(npix):
                ix = xid==j
                if np.sum(ix):
                    lam[ix] /= np.sum(lam[ix]**2)**.5
                    X0[j, :] = lam[ix] @ X[ix, :]
            A = S @ X0
            if self.constraints>=2:
                nA      = np.sum(A**2, axis=1)**.5
                nA      /= self.vscale[:nc]
                A        /= nA[:, np.newaxis]

            AtS     = A.T @ S
            if self.mode=='parallel':
                X = Xall[t%2]
            cv      = X @ AtS
            vnorm   = np.sum(AtS**2, axis=0)

            cmap    = assign_neurons2(vnorm, cv)
            cmax    = np.amax(cmap, axis=1)
            xid     = np.argmax(cmap, axis=1)
            lam     = np.sqrt(cmax / vnorm[xid])

            E[t]    = np.nanmean(cmax)/np.nanmean(xnorm)

            if self.mode is 'parallel':
                cmapx[t%2] = cmap
            if t%10==0:
                if self.verbose:
                    print('%2.2fs    %2.0d        %4.4f      %d'%(time.time()-tic, t, E[t], nc))
        if self.verbose:
            print('%2.2fs   final      %4.4f'%(time.time()-tic, E[t]))

        if self.mode is 'parallel':
            iclustup1 = upsample_grad(np.sqrt(cmapx[0]), dims, nclust).T
            iclustup2 = upsample_grad(np.sqrt(cmapx[1]), dims, nclust).T
            iclustup = np.concatenate((iclustup1[np.newaxis, :, :], iclustup2[np.newaxis, :, :]), axis=0)
            isort = np.argsort(iclustup2[:,0])
            self.cmap = cmapx
        else:
            if self.n_components==2:
                iclustup        = upsample_grad(np.sqrt(cmap), dims, nclust).T
                #iclustup, cmax = upsample(np.sqrt(cmap), dims, nclust, 10)
                self.embedding0 = iclustup
            else:
                iclustup, cmax = upsample(np.sqrt(cmap), dims, nclust, 10)
            isort = np.argsort(iclustup[:,0])
            self.cmap = cmap
        if self.verbose:
            print('%2.2fs upsampled    %4.4f'%(time.time()-tic, E[t]))
        self.E = E
        self.S = S
        self.A = A
        self.lam = lam
        self.X0 = X0
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
