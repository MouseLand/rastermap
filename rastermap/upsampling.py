import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA


def quadratic_upsampling(X, cc, x_m, y_m):
    n_X = x_m.shape[0]
    n_samples = X.shape[0]

    cbest = cc.argmax(axis=0)

    ibest, jbest = np.unravel_index(cbest, (n_X, n_X))
    imin = np.maximum(0, ibest-1)
    imin[n_X - (ibest+2) < 0] -= 1
    jmin = np.maximum(0, jbest-1)
    jmin[n_X - (jbest+2) < 0] -= 1
    icent, jcent = imin+1, jmin+1

    igrid, jgrid = np.meshgrid(np.arange(0,3), np.arange(0,3), indexing='ij')
    igrid, jgrid = igrid.flatten(), jgrid.flatten()
    iinds = igrid + imin[:,np.newaxis]
    jinds = jgrid + jmin[:,np.newaxis]
    igrid = igrid.astype(np.float32) - 1.
    jgrid = jgrid.astype(np.float32) - 1.

    cinds = np.ravel_multi_index((iinds, jinds), (n_X, n_X))
    C = cc[cinds, np.tile(np.arange(0, n_samples)[:,np.newaxis], (1, 9))]
    IJ = np.stack((np.ones_like(igrid), igrid**2 + jgrid**2, igrid, jgrid), axis=1)
    A = np.linalg.solve(IJ.T @ IJ, IJ.T @ C.T)
    xmax = np.clip(-A[2] / (2*A[1]), -1, 1)
    ymax = np.clip(-A[3] / (2*A[1]), -1, 1)

    # put in original space
    xdelta = np.diff(x_m[:,0]).mean()
    ydelta = np.diff(y_m[0]).mean()
    xmax = xmax*xdelta + x_m[icent,0]
    ymax = ymax*ydelta + y_m[0,jcent]
    #xmax += icent
    #ymax += jcent

    Y = np.stack((xmax, ymax), axis=1)
    return Y

def grid_upsampling(X, X_nodes, Y_nodes, n_X=41, n_neighbors=50):
    n_X = 41
    x_m = np.linspace(Y_nodes[:,0].min(), Y_nodes[:,0].max(), n_X)
    y_m = np.linspace(Y_nodes[:,1].min(), Y_nodes[:,1].max(), n_X)

    x_m, y_m = np.meshgrid(x_m, y_m, indexing='ij')
    xy = np.vstack((x_m.flatten(), y_m.flatten()))

    ds = (xy[0][:,np.newaxis] - Y_nodes[:,0])**2 + (xy[1][:,np.newaxis] - Y_nodes[:,1])**2 
    isort = np.argsort(ds, 1)[:,:n_neighbors]
    nraster = xy.shape[1]
    Xrec = np.zeros((nraster, X_nodes.shape[1]))
    for j in range(nraster):
        ineigh = isort[j]
        dists = ds[j, ineigh]
        w = np.exp(-dists / dists[7])
        M, N = X_nodes[ineigh], Y_nodes[ineigh]
        N = np.concatenate((N, np.ones((n_neighbors,1))), axis=1)
        R = np.linalg.solve((N.T * w) @ N, (N.T * w) @ M)
        Xrec[j] = xy[:,j] @ R[:2] + R[-1]

    Xrec = Xrec / (Xrec**2).sum(1)[:,np.newaxis]**.5
    cc = Xrec @ zscore(X, 1).T
    cc = np.maximum(0, cc)
    imax = np.argmax(cc, 0)
    Y = xy[:, imax].T

    return Y, cc, x_m, y_m


def LLE_upsampling(X, X_nodes, Y_nodes, n_neighbors=10, LLE = 1):
    """ X is original space points, X_nodes nodes in original space, Y_nodes nodes in embedding space """
    e_dists = ((Y_nodes[:,:,np.newaxis] - Y_nodes.T)**2).sum(axis=1)
    cc = -np.sum(X_nodes**2, 1)[:,np.newaxis]  - np.sum(X**2, 1) + 2 * X_nodes @ X.T 
    y = np.zeros((X.shape[0],2))
    for i in range(X.shape[0]):
        x = X[i]
        ineigh0 = cc[:,i].argmax() #cc[:,i].argsort()[::-1][:n_neighbors]
        ineigh = e_dists[ineigh0].argsort()[:n_neighbors]
        if LLE:
            z = X_nodes[ineigh] - x
            G = z @ z.T
            alpha = 1e-8
            w = np.linalg.solve(G + alpha*np.eye(n_neighbors), np.ones(n_neighbors, np.float32))
        else:
            w = np.linalg.solve(X_nodes[ineigh] @ X_nodes[ineigh].T, X_nodes[ineigh] @ x)
        w /= w.sum()
        y[i] = w @ Y_nodes[ineigh]
    return y

def PCA_upsampling(X, X_nodes, Y_nodes):
    n_samples, n_features = X.shape
    n_nodes = X_nodes.shape[0]
    n_components = Y_nodes.shape[1]

    cc = -np.sum(X_nodes**2, 1)[:,np.newaxis]  - np.sum(X**2, 1) + 2 * X_nodes @ X.T 
    inode = cc.argmax(axis=0)
    Y = np.zeros((n_samples, 2))
    for n in range(n_nodes):
        pts = X[inode==n]
        delta = PCA(n_components=2).fit_transform(pts)
        Y[inode==n] = Y_nodes[n] +  1e-2 * delta

    return Y

def knn_upsampling(X, X_nodes, Y_nodes, n_neighbors=10):
    n_samples = X.shape[0]
    Ndist = np.sum(X_nodes**2, 1)[:,np.newaxis]  + np.sum(X**2, 1) - 2 * X_nodes @ X.T
    inds_k = Ndist.argsort(axis=0)[:n_neighbors]
    Ndist_k = np.sort(Ndist, axis=0)[:n_neighbors]
    sigma = Ndist_k[0]
    w = np.exp(-1 * Ndist_k / sigma)
    w /= w.sum(axis=0)
    Y_knn = (w[...,np.newaxis] * Y_nodes[inds_k]).sum(axis=0)
    return Y_knn

def subspace_upsampling2(X, X_nodes, Y_nodes, n_neighbors=10):
    e_dists = ((Y_nodes[:,:,np.newaxis] - Y_nodes.T)**2).sum(axis=1)
    cc = -np.sum(X_nodes**2, 1)[:,np.newaxis]  - np.sum(X**2, 1) + 2 * X_nodes @ X.T 
    #cc = zscore(X_nodes,axis=1) @ zscore(X,axis=1).T

    n_samples, n_features = X.shape
    n_nodes = X_nodes.shape[0]
    n_components = Y_nodes.shape[1]

    Y = np.zeros((n_samples, n_components))
    for n in range(n_nodes):
        ineigh = e_dists[n].argsort()[:n_neighbors]
        # min || M  - (a * N @ R + b) ||
        M, N = X_nodes[ineigh], Y_nodes[ineigh]
        ones = np.ones((n_neighbors,1))
        a = 1
        b = 0
        for k in range(1):
            cov = M.T @ N
            model = PCA(n_components=n_components).fit(cov)
            vv = model.components_.T
            uv = (cov @ vv.T) / model.singular_values_
            R = vv @ uv.T

            a = ((M.T @ (N @ R)).sum() - (R.T @ N.T * b).sum()) / (R.T @ N.T @ N @ R).sum()
            
            R1 = np.linalg.solve(N.T @ N, N.T @ M)
            print( ((M - N @ R)**2).sum(), ((M - (a *N @ R + b))**2).sum(), ((M - N @ R1)**2).sum())




    return Y 

if 0:
    Y = np.zeros((n_nodes, n_samples, n_components))
    rerr = np.zeros((n_nodes, n_samples))
    for n in range(n_nodes):
        ineigh = e_dists[n].argsort()[:n_neighbors]
        M, N = X_nodes[ineigh], Y_nodes[ineigh]
        N = np.concatenate((N, np.ones((n_neighbors,1))), axis=1)
        R = np.linalg.solve(N.T @ N, N.T @ M)
        Xz = X.copy() - R[2]
        Y_est[n] = np.linalg.solve(R[:2] @ R[:2].T, R[:2] @ Xz.T).T
        rerr[n] = ((Xz - Y_est[n] @ R[:2])**2).sum(axis=1)
        
    # Y = Y_est[rerr.argmin(axis=0), np.arange(0, n_samples)]
    Y = Y_est[cc.argmax(axis=0), np.arange(0, n_samples)]
    #return Y