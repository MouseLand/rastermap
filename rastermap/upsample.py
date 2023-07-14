"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.stats import zscore


def grid_upsampling(X, X_nodes, Y_nodes, n_X, n_neighbors=50, e_neighbor=1):
    e_neighbor = min(n_neighbors - 1, e_neighbor)
    xy = []
    n_clusters = Y_nodes.max() + 1
    grid_upsample = np.round(n_X / n_clusters)
    for i in range(Y_nodes.shape[1]):
        xy.append(np.arange(0, n_clusters, 1. / grid_upsample))
        #xy.append(np.linspace(Y_nodes[:,i].min(), Y_nodes[:,i].max(), n_X))
    if Y_nodes.shape[1] == 2:
        x_m, y_m = np.meshgrid(xy[0], xy[1], indexing="ij")
        xy = np.vstack((x_m.flatten(), y_m.flatten()))
    else:
        xy = xy[0][np.newaxis, :]

    ds = np.zeros((xy.shape[1], Y_nodes.shape[0]))
    n_components = len(xy)
    for i in range(len(xy)):
        ds += (xy[i][:, np.newaxis] - Y_nodes[:, i])**2
    isort = np.argsort(ds, 1)[:, :n_neighbors]
    nraster = xy.shape[1]
    Xrec = np.zeros((nraster, X_nodes.shape[1]))
    for j in range(nraster):
        ineigh = isort[j]
        dists = ds[j, ineigh]
        w = np.exp(-dists / dists[e_neighbor])
        M, N = X_nodes[ineigh], Y_nodes[ineigh]
        N = np.concatenate((N, np.ones((n_neighbors, 1))), axis=1)
        R = np.linalg.solve((N.T * w) @ N, (N.T * w) @ M)
        Xrec[j] = xy[:, j] @ R[:-1] + R[-1]

    Xrec = Xrec / (Xrec**2).sum(1)[:, np.newaxis]**.5
    cc = Xrec @ zscore(X, 1).T
    cc = np.maximum(0, cc)
    imax = np.argmax(cc, 0)
    Y = xy[:, imax].T

    return Y, cc, xy, Xrec
