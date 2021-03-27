import numpy as np
from numba import njit, jit, float32, int32, boolean, int64, vectorize, prange
from numba.types import Tuple
import itertools, time
from sklearn.cluster import KMeans
from scipy.stats import zscore

def kmeans(X, n_clusters=100):
    X_norm = 1 #(1e-10 + (X**2).sum(axis=0)**.5)
    model = KMeans(n_init=1, n_clusters=n_clusters, random_state=0).fit(X / X_norm)
    X_nodes = model.cluster_centers_ * X_norm
    X_nodes = X_nodes / (1e-10 + ((X_nodes**2).sum(axis=1))[:,np.newaxis])**.5  
    cc = X @ X_nodes.T 
    imax = cc.argmax(axis=1)
    return X_nodes, imax

def create_ND_basis(dims, nclust, K, flag=True):
    # recursively call this function until we fill out S
    if dims==1:
        xs = np.arange(0,nclust)
        S = np.ones((K, nclust), 'float64')
        for k in range(K):
            if flag:
                S[k, :] = np.sin(np.pi + (k+1)%2 * np.pi/2 + 2*np.pi/nclust * (xs+0.5) * int((1+k)/2))
            else:
                S[k, :] = np.cos(np.pi/nclust * (xs+0.5) * k)
        S /= np.sum(S**2, axis = 1)[:, np.newaxis]**.5
        if flag:
            fxx = np.floor((np.arange(K)+1)/2).astype('int')
        else:
            fxx = np.arange(K).astype('int')
    else:
        S0, fy = create_ND_basis(dims-1, nclust, K, flag)
        Kx, fx = create_ND_basis(1, nclust, K, flag)
        S = np.zeros((S0.shape[0], K, S0.shape[1], nclust), np.float64)
        fxx = np.zeros((S0.shape[0], K))
        for kx in range(K):
            for ky in range(S0.shape[0]):
                S[ky,kx,:,:] = np.outer(S0[ky, :], Kx[kx, :])
                fxx[ky,kx] = ((0+fy[ky])**2 + (0+fx[kx])**2)**0.5
                #fxx[ky,kx] = fy[ky] + fx[kx]
                #fxx[ky,kx] = max(fy[ky], fx[kx]) + min(fy[ky], fx[kx])/1000.
        S = np.reshape(S, (K*S0.shape[0], nclust*S0.shape[1]))
    fxx = fxx.flatten()
    ix = np.argsort(fxx)
    S = S[ix, :]
    fxx = fxx[ix]
    return S, fxx

@njit('float32 (float32[:,:,:,:], float32[:,:,:,:])', nogil=True)
def elementwise_mult_sum(x, y):
    return (x * y).sum()

@njit('int64[:] (int64, int64, int64, int64)', nogil=True)
def shift_inds(i0, i1, inode, n_nodes):
    """ shift segment from i0->i1 to position inode"""
    n_seg = i1 - i0 
    l_seg = inode - i0 
    if l_seg>=0:
        inds = np.concatenate((np.arange(i0), 
                               np.arange(i1,i1+l_seg),  
                               np.arange(i0, i1), 
                               np.arange(i1+l_seg, n_nodes))) 
    else:
        inds = np.concatenate((np.arange(inode), 
                               np.arange(i0, i1), 
                               np.arange(inode, i0), 
                               np.arange(i1,n_nodes)))
    
    return inds

@njit('(float32[:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64)', nogil=True)
def shift_matrix_forward(cc, i0, i1, inode, j0, j1, jnode, move_order, n_nodes):
    
    ishift = shift_inds(i0, i1, inode, n_nodes)
    jshift = shift_inds(j0, j1, jnode, n_nodes)

    ilength = i1 - i0
    jlength = j1 - j0

    cc_new = cc.copy()
    if move_order==0:
        cc_new[j0:j1] = cc_new[j0:j1][:,ishift]
        cc_new[:,:,j0:j1] = cc_new[:,:,j0:j1][:,:,:,ishift]
        cc_new[:,inode:inode+ilength] = cc_new[:,inode:inode+ilength][jshift]
        cc_new[:,:,:,inode:inode+ilength] = cc_new[:,:,:,inode:inode+ilength][:,:,jshift]
    else:
        cc_new[:,i0:i1] = cc_new[:,i0:i1][jshift]
        cc_new[:,:,:,i0:i1] = cc_new[:,:,:,i0:i1][:,:,jshift]
        cc_new[jnode:jnode+jlength] = cc_new[jnode:jnode+jlength][:,ishift]
        cc_new[:,:,jnode:jnode+jlength] = cc_new[:,:,jnode:jnode+jlength][:,:,:,ishift]

    return cc_new, ishift, jshift

@njit('(float32[:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64, int64)', nogil=True)
def shift_matrix(cc, i0, i1, inode, j0, j1, jnode, move_order, n_nodes, ordering):
    cc_new, ishift, jshift = shift_matrix_forward(cc, i0, i1, inode, j0, j1, jnode, move_order, n_nodes)
    ilength = i1 - i0
    jlength = j1 - j0
    if ordering==1:
        cc_new[jnode:jnode+jlength, inode:inode+ilength] = cc_new[jnode:jnode+jlength, inode:inode+ilength][:,::-1]
        cc_new[:,:,jnode:jnode+jlength, inode:inode+ilength] = cc_new[:,:,jnode:jnode+jlength, inode:inode+ilength][:,:,:,::-1]
        ishift[inode:inode+ilength] = ishift[inode:inode+ilength][::-1]
    elif ordering==2:
        cc_new[jnode:jnode+jlength, inode:inode+ilength] = cc_new[jnode:jnode+jlength, inode:inode+ilength][::-1]
        cc_new[:,:,jnode:jnode+jlength, inode:inode+ilength] = cc_new[:,:,jnode:jnode+jlength, inode:inode+ilength][:,:,::-1]
        jshift[jnode:jnode+jlength] = jshift[jnode:jnode+jlength][::-1]
    elif ordering==3:
        cc_new[jnode:jnode+jlength, inode:inode+ilength] = cc_new[jnode:jnode+jlength, inode:inode+ilength][::-1, ::-1]
        cc_new[:,:,jnode:jnode+jlength, inode:inode+ilength] = cc_new[:,:,jnode:jnode+jlength, inode:inode+ilength][:,:,::-1,::-1]
        ishift[inode:inode+ilength] = ishift[inode:inode+ilength][::-1]
        jshift[jnode:jnode+jlength] = jshift[jnode:jnode+jlength][::-1]

    return cc_new, ishift, jshift

#@njit('(float32[:,:], int32[:])', nogil=True)
def shift_rows(cc, ishift):
    return cc[ishift]

@njit('float32[:] (float32[:,:,:,:], int64, int64, int64, int64, int64, int64, int64, int64, float32[:,:,:,:])', nogil=True)
def new_correlation(cc, i0, i1, inode, j0, j1, jnode, move_order, n_nodes, BBt):
    """ compute correlation change of moving segment i0:i1+1 to position inode
    inode=-1 is at beginning of sequence
    """
    
    cc_new = shift_matrix_forward(cc, i0, i1, inode, j0, j1, jnode, move_order, n_nodes)[0]
    ilength = i1 - i0
    jlength = j1 - j0
    
    corr_new = elementwise_mult_sum(cc_new, BBt)
    ordering = 0

    if ilength > 1:
         cc0 = cc_new.copy()
         cc0[jnode:jnode+jlength, inode:inode+ilength] = cc0[jnode:jnode+jlength, inode:inode+ilength][:,::-1]
         cc0[:,:,jnode:jnode+jlength, inode:inode+ilength] = cc0[:,:,jnode:jnode+jlength, inode:inode+ilength][:,:,:,::-1]
         corr_new0 = elementwise_mult_sum(cc0, BBt)
         if corr_new0 > corr_new:
             corr_new = corr_new0 
             ordering = 1

    if jlength > 1:
        cc0 = cc_new.copy()
        cc0[jnode:jnode+jlength, inode:inode+ilength] = cc0[jnode:jnode+jlength, inode:inode+ilength][::-1]
        cc0[:,:,jnode:jnode+jlength, inode:inode+ilength] = cc0[:,:,jnode:jnode+jlength, inode:inode+ilength][:,:,::-1]
        corr_new0 = elementwise_mult_sum(cc0, BBt)
        if corr_new0 > corr_new:
            corr_new = corr_new0 
            ordering = 2

    if jlength > 1 and ilength>1:
        cc0 = cc_new.copy()
        cc0[jnode:jnode+jlength, inode:inode+ilength] = cc0[jnode:jnode+jlength, inode:inode+ilength][::-1,::-1]
        cc0[:,:,jnode:jnode+jlength, inode:inode+ilength] = cc0[:,:,jnode:jnode+jlength, inode:inode+ilength][:,:,::-1,::-1]
        corr_new0 = elementwise_mult_sum(cc0, BBt)
        if corr_new0 > corr_new:
            corr_new = corr_new0 
            ordering = 3

    corr_out = np.zeros(2, np.float32)
    corr_out[0] = corr_new
    corr_out[1] = ordering
    return corr_out

@njit('(float32[:,:,:,:], int32[:,:,:], int64, int64, int64, int64, int64, int64[:], int64[:], int64[:], int64[:], int64[:], int64[:], float32[:,:,:,:], boolean)', nogil=True, parallel=True)
def tsp_fast(cc, ind_shift, n_iter, n_nodes, n_len, n_test, n_skip, 
             ilengths, i0s, inodes, jlengths, j0s, jnodes, BBt, verbose):
    print(n_len * n_nodes**2 * n_test**2 * 2)
    corr_orig = 0
    corr_change_seg = -np.inf * np.ones(n_len * n_nodes**2 * n_test**2 * 2, np.float32)
    ordering_seg = -np.inf * np.ones(n_len *  n_nodes**2 * n_test**2 * 2, np.float32)
    seg_len = np.zeros(n_iter)
    for k in range(n_iter):
        improved = False
        if corr_orig==0:
            corr_orig = elementwise_mult_sum(cc, BBt) 
        for tli in range(0, (n_nodes-1)//n_len):
            for tlj in range(0, (n_nodes-1)//n_len):
                corr_change_seg[:] = -np.inf
                for ix in prange(0, n_len * n_nodes**2 * n_test**2 * 2):
                    move_order = ix%2
                    ilength, i0, inode = ilengths[ix], i0s[ix], inodes[ix]
                    ilength += tli*n_len
                    i1 = (i0 + ilength) 
                    inode += k%n_skip
                    jlength, j0, jnode = jlengths[ix], j0s[ix], jnodes[ix]
                    jlength += tlj*n_len
                    j1 = (j0 + jlength)
                    jnode += k%n_skip
                    
                    if (( i1 <= n_nodes and inode + ilength <= n_nodes) and 
                        ( j1 <= n_nodes and jnode + jlength <= n_nodes)):
                        new_corr = new_correlation(cc, i0, i1, inode, j0, j1, jnode, move_order, n_nodes, BBt)
                        ordering = new_corr[1]
                        new_corr = new_corr[0]
                        corr_change = new_corr - corr_orig
                        corr_change_seg[ix] = corr_change
                        ordering_seg[ix] = ordering

                ix = corr_change_seg.argmax()
                corr_change = corr_change_seg[ix]
                if corr_change > 1e-3:
                    improved = True
                    break 
            if corr_change > 1e-3:
                improved = True
                break 

        if not improved:
            break
        else:
            ordering = ordering_seg[ix]
            move_order = ix%2
            ilength, i0, inode = ilengths[ix], i0s[ix], inodes[ix]
            ilength += tli*n_len
            i1 = (i0 + ilength) 
            inode += k%n_skip
            jlength, j0, jnode = jlengths[ix], j0s[ix], jnodes[ix]
            j1 = (j0 + jlength) 
            jnode += k%n_skip
            jlength += tlj*n_len
            if corr_change > 1e-3:
                # move segment
                cc, ishift, jshift = shift_matrix(cc, i0, i1, inode, j0, j1, jnode, move_order, n_nodes, ordering)
                if move_order==0:
                    ind_shift[j0:j1] = ind_shift[j0:j1][:,ishift]
                    ind_shift[:,inode:inode+ilength] = ind_shift[:,inode:inode+ilength][jshift]
                else:
                    ind_shift[:,i0:i1] = ind_shift[:,i0:i1][jshift]
                    ind_shift[jnode:jnode+jlength] = ind_shift[jnode:jnode+jlength][:,ishift]
                
                corr_new = elementwise_mult_sum(cc, BBt)
                if verbose:
                    print(k, ordering, i0, i1, inode, j0, j1, jnode, corr_change, corr_new - corr_orig, corr_new)
                corr_orig = corr_new
                seg_len[k] = i1 - i0
    
    return cc, ind_shift, seg_len


def travelling_salesman(cc, n_iter=400, alpha=1.0, n_len=None, n_skip=None, greedy=False, verbose=False):
    """ matches correlation matrix cc to B@B.T basis functions """
    n_nodes = cc.shape[0]
    if n_skip is None:
        n_skip = max(1, n_nodes // 30)
    cc = cc.astype(np.float32)

    if n_len is None:
        n_len = n_nodes-2

    n_components = 1

    if alpha > 0:
        basis = .5
        B, plaw = create_ND_basis(n_components, n_nodes, int(n_nodes*basis), flag=False)
        B = B[1:]
        plaw = plaw[1:]
        n_basis, n_nodes = B.shape
        B /= plaw[:,np.newaxis] ** (alpha/2)
        B_norm = (B**2).sum(axis=0)**0.5
        B = B / B_norm
        BBt = B.T @ B #/ (B**2).sum(axis=1)

        x = np.arange(0, 1.0, 1.0/(n_nodes))
        x = np.array(np.meshgrid(x, x)).reshape(2,-1).T
        BBt = compute_BBt(x, x)
        BBt -= np.diag(np.diag(BBt))
        BBt = BBt.reshape(n_nodes,n_nodes,n_nodes,n_nodes)
    else:
        BBt = np.ones((n_nodes, n_nodes))
        BBt = np.tril(np.triu(BBt, -1), 1)

    n_iter = np.int64(n_iter)
    ilengths, i0s, inodes, jlengths, j0s, jnodes = np.meshgrid(np.arange(1, n_len+1), 
                                                                np.arange(0, n_nodes),
                                                                np.arange(0, n_nodes, n_skip),
                                                                np.arange(1, n_len+1), 
                                                                np.arange(0, n_nodes),
                                                                np.arange(0, n_nodes, n_skip),
                                                                indexing='ij')
    ilengths, i0s, inodes = ilengths.flatten(), i0s.flatten(), inodes.flatten()
    jlengths, j0s, jnodes = jlengths.flatten(), j0s.flatten(), jnodes.flatten()
    #ind_shift = np.array(np.meshgrid(np.arange(0, n_nodes, 1, np.int32), 
    #                                 np.arange(0, n_nodes, 1, np.int32),
    #                                 indexing='ij')).transpose(1,2,0)
    ind_shift = np.stack((np.arange(0,n_nodes**2, 1, np.int32).reshape(n_nodes, n_nodes), 
                         np.arange(0,n_nodes**2, 1, np.int32).reshape(n_nodes, n_nodes).T), axis=-1)
    n_test = len(np.arange(0, n_nodes, n_skip))
    cc, inds, seg_len = tsp_fast(cc, ind_shift, n_iter, n_nodes, n_len, n_test, n_skip, 
                                 ilengths, i0s, inodes,
                                 jlengths, j0s, jnodes,
                                 BBt.astype(np.float32), verbose)
    if n_skip > 1:
        n_skip = 1
        ilengths, i0s, inodes, jlengths, j0s, jnodes = np.meshgrid(np.arange(1, n_len+1), 
                                                                np.arange(0, n_nodes),
                                                                np.arange(0, n_nodes, n_skip),
                                                                np.arange(1, n_len+1), 
                                                                np.arange(0, n_nodes),
                                                                np.arange(0, n_nodes, n_skip),
                                                                indexing='ij')
        ilengths, i0s, inodes = ilengths.flatten(), i0s.flatten(), inodes.flatten()
        jlengths, j0s, jnodes = jlengths.flatten(), j0s.flatten(), jnodes.flatten()
        cc, inds, seg_len2 = tsp_fast(cc, inds, n_iter, n_nodes, n_len, n_test, n_skip, 
                                 ilengths, i0s, inodes,
                                 jlengths, j0s, jnodes,
                                 BBt.astype(np.float32), verbose)
        #inds = inds[inds2]

    return cc, inds, seg_len, BBt

def compute_BBt(xi, yi, alpha=1e3):
    BBt = - np.log(1e-10 + ((xi[:,np.newaxis] - yi)**2).sum(axis=-1)**0.5)
    #BBt = 1 / (1 + alpha * (xi[:,np.newaxis] - yi)**2)**0.5
    return BBt

def cluster_split_and_sort(U, n_clusters=50, nc=25, n_splits=4, alpha=1.0, sticky=True):
    U_nodes, imax = kmeans(U, n_clusters=n_clusters)
    cc = U_nodes @ U_nodes.T
    cc,inds,seg_len = travelling_salesman(cc, verbose=False, alpha=alpha)
    U_nodes = U_nodes[inds]
    
    n_PCs = U_nodes.shape[1]
    ineurons = (U @ U_nodes.T).argmax(axis=1)
    for k in range(n_splits):
        U_nodes_new = np.zeros((0, n_PCs))
        n_nodes = U_nodes.shape[0]
        if not sticky:
            ineurons = (U @ U_nodes.T).argmax(axis=1)
        ineurons_new = -1*np.ones(U.shape[0], np.int64)
        for i in range(n_nodes//nc):
            ii = np.arange(n_nodes)
            node_set = np.logical_and(ii>=i*nc, ii<(i+1)*nc)
            in_set = np.logical_and(ineurons>=i*nc, ineurons<(i+1)*nc)
            U_nodes0, ineurons_set = kmeans(U[in_set], n_clusters=2*nc)
            cc = U_nodes0 @ U_nodes0.T
            cc_add = U_nodes0 @ U_nodes[~node_set].T
            ifrac = node_set.mean()
            x = np.linspace(i*nc/n_nodes, (i+1)*nc/n_nodes, 2*nc+1)[:-1]
            y = np.linspace(0, 1, n_nodes+1)[:-1][~node_set]
            BBt = compute_BBt(x, x)
            BBt -= np.diag(np.diag(BBt))

            BBt_add = compute_BBt(x, y)

            cc_out,inds,seg_len = matrix_matching(cc, BBt, 
                                                    cc_add, BBt_add, 
                                                    verbose=False)#cc.shape[0]-25)
            U_nodes0 = U_nodes0[inds]
            ineurons_new[in_set] = 2*nc*i + (U[in_set] @ U_nodes0.T).argmax(axis=1)
            U_nodes_new = np.vstack((U_nodes_new, U_nodes0))
        n_nodes = U_nodes_new.shape[0]
        U_nodes = U_nodes_new.copy()
        ineurons = ineurons_new.copy()
    Y_nodes = np.arange(0, U_nodes.shape[0])[:,np.newaxis]
    
    if not sticky:
        ineurons = (U @ U_nodes.T).argmax(axis=1)
    return U_nodes, Y_nodes, ineurons