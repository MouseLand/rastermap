import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA


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

def subspace_upsampling(X, X_nodes, Y_nodes, n_neighbors=10):
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