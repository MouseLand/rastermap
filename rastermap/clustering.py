import numpy as np
from numba import njit, jit, float32, int32, boolean, int64, vectorize, prange
from numba.types import Tuple
import itertools, time
from sklearn.cluster import KMeans
from scipy.stats import zscore

def kmeans(X, n_clusters=100):
    X_norm = 1 #(1e-10 + (X**2).sum(axis=0)**.5)
    model = KMeans(n_init=1, n_clusters=n_clusters, init = 'random', random_state=0).fit(X / X_norm)
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

@njit('float32 (float32[:,:], float32[:,:])', nogil=True)
def elementwise_mult_sum(x, y):
    return (x * y).sum()

@njit('int64[:] (int64, int64, int64, int64)', nogil=True)
def shift_inds2(i0, i1, inode, n_nodes):
   """ shift segment from i0->i1 to position inode"""
   n_seg = i1 + 1 - i0
   l_seg = inode + 1 - i0
   if l_seg>=0:
       inds = np.concatenate((np.arange(i0),
                              np.arange(i1+1,i1+1+l_seg),
                              np.arange(i0, i1+1),
                              np.arange(i1+1+l_seg, n_nodes)))
   else:
       inds = np.concatenate((np.arange(inode+1),
                              np.arange(i0, i1+1),
                              np.arange(inode+1, i0),
                              np.arange(i1+1,n_nodes)))
   return inds

@njit('int32[:] (int64, int64, int64, int64, int64)', nogil=True)
def shift_inds(i0, i1, inode, isforward, n_nodes):
    """ shift segment from i0->i1 to position inode"""
    n_seg = i1 - i0 + 1
    inds = np.arange(0, n_nodes, 1, np.int32)
    inds0 = inds.copy()
    if inode > i0:
        inds[i0 : i0 + inode - i1] = inds0[i1+1 : inode+1]
        seg_pos = inode - n_seg
    else:
        inds[inode + 1 + n_seg : i0 + n_seg] = inds0[inode + 1 : i0]
        seg_pos = inode
    if isforward:
        inds[seg_pos+1 : seg_pos+1 + n_seg] = inds0[i0 : i1+1]
    else:
        inds[seg_pos+1 : seg_pos+1 + n_seg] = inds0[i0 : i1+1][::-1]
    return inds

@njit('(float32[:,:], int32[:])', nogil=True)
def shift_matrix(cc, ishift):
    return cc[ishift][:,ishift]

@njit('(float32[:,:], int32[:])', nogil=True)
def shift_rows(cc, ishift):
    return cc[ishift]

@njit('float32 (float32[:,:], int64, int64, int64, int64, int64, float32[:,:], int64, int64)', nogil=True)
def new_correlation(cc, i0, i1, inode, n_nodes, isforward, BBt, edge_min, edge_max):
    """ compute correlation change of moving segment i0:i1+1 to position inode
    inode=-1 is at beginning of sequence
    """
    ishift = shift_inds(i0, i1, inode, isforward, edge_max - edge_min)
    iinds = np.arange(0, n_nodes).astype(np.int32)
    iinds[edge_min : edge_max] = ishift + edge_min
    cc2 = shift_matrix(cc, iinds)
    corr_new = elementwise_mult_sum(cc2, BBt)
    return corr_new

@njit('float32 (float32[:,:], float32[:,:], int64, int64, int64, int64, int64, float32[:,:], float32[:,:])', nogil=True)
def new_correlation_sub(cc, cc_add, i0, i1, inode, n_nodes, isforward, BBt, BBt_add):
    """ compute correlation change of moving segment i0:i1+1 to position inode
    inode=-1 is at beginning of sequence
    """
    ishift = shift_inds(i0, i1, inode, isforward, n_nodes)
    cc2 = shift_matrix(cc, ishift)
    cc_add2 = shift_rows(cc_add, ishift)
    corr_new = elementwise_mult_sum(cc2, BBt) + 4*elementwise_mult_sum(cc_add2, BBt_add)
    return corr_new

@njit('(float32[:,:], int32, int32, int32, float32[:,:], float32)', nogil=True)
def test_segment(cc, i0, i1, n_nodes, BBt, corr_orig):
    """ test movements of segment i0 and i1 and return optimal """
    test_nodes = np.arange(-1, n_nodes, 1, np.int32)
    n_test = len(test_nodes)
    iinds = np.arange(0, n_nodes, 1, np.int32)
    if n_test > 0:
        corr_changes = np.inf * np.ones((n_test))
        forward_min = np.zeros(n_test, np.int32)
        for ip in range(len(test_nodes)):
            inode = test_nodes[ip]
            if inode < i0 or inode > i1+1:
                for isforward in [0,1]:
                    new_dist = new_correlation(cc, i0, i1, inode, n_nodes, isforward, BBt, 0, n_nodes)
                    if not isforward:
                        corr_changes[ip] = corr_orig - new_dist
                    else:
                        if corr_orig - new_dist < corr_changes[ip]:
                            corr_changes[ip] = corr_orig - new_dist
                            forward_min[ip] = 1
        ip = corr_changes.argmin()
        corr_change_min = corr_changes[ip]
        inode = test_nodes[ip]
        isforward = forward_min[ip]
        return inode, corr_change_min, isforward
    else:
        return 0, 0, 0

@njit('(float32[:,:], int32, int32, float32[:,:], int32[:,:])', nogil=True, parallel=True)
def tsp_greedy(cc, n_iter, n_nodes, BBt, test_segs):
    inds = np.arange(0, n_nodes)
    n_segs = len(test_segs)
    seg_len = np.ones(n_iter)
    for k in range(n_iter):
        params = np.inf * np.ones((n_segs, 5), np.float32)
        corr_orig = (cc * BBt).sum()
        for ix in prange(len(test_segs)):
            i0, i1 = test_segs[ix]
            inode, corr_change_min, isforward = test_segment(
                        cc, i0, i1, n_nodes, BBt, corr_orig)
            params[ix] = np.array([i0, i1, inode, corr_change_min, isforward])

        ix = params[:,3].argmin()
        i0, i1, inode, corr_change_min, isforward = params[ix]
        i0, i1, inode = int(i0), int(i1), int(inode)
        n_seg = i1 - i0 + 1
        if corr_change_min < -1e-3:
            print(i0, i1, inode)
            true_dists = (cc * BBt).sum()

            # move segment
            ishift = shift_inds(i0, i1, inode, isforward, n_nodes)
            inds = inds[ishift]
            cc = shift_matrix(cc, ishift)#np.ix_(ishift, ishift)]

            new_dists = elementwise_mult_sum(cc, BBt)
            print(k, corr_change_min, new_dists - true_dists, new_dists)
            seg_len[k] = i1 - i0
        else:
            break
    return cc, inds, seg_len

@njit('(float32[:,:], int64, int64, int64, float32[:,:], int64, int64, boolean)', nogil=True, parallel=True)
def tsp_fast(cc, n_iter, n_nodes, n_skip, BBt, edge_min, edge_max, verbose):
    n_nodes_check = edge_max - edge_min
    inds = np.arange(0, n_nodes_check).astype(np.int32)
    iinds = np.arange(0, n_nodes).astype(np.int32)
    seg_len = np.ones(n_iter)
    n_len = 2
    n_test = n_nodes_check//n_skip + 1
    test_lengths = np.arange(0, ((n_nodes_check - 1)//n_len)*n_len).reshape(-1,n_len)
    corr_orig = 0

    corr_change_seg = -np.inf * np.ones(n_len * n_nodes_check * n_test, np.float32)
    isforward_seg = -np.inf * np.ones(n_len *  n_nodes_check * n_test, np.float32)
    for k in range(n_iter):
        improved = False
        if corr_orig==0:
            corr_orig = elementwise_mult_sum(cc, BBt)
        for tl in range(len(test_lengths)):
            corr_change_seg[:] = -np.inf
            for ix in prange(0, n_len*n_nodes_check*n_test):
                seg_length = (ix // n_test // n_nodes_check) + test_lengths[tl,0]
                i0 = ((ix // n_test) % n_nodes_check)
                inode = (ix % n_test)*n_skip - 1 + k%n_skip
                i1 = (i0 + seg_length)
                if i1 < n_nodes_check and (inode < i0 or inode > i1+1) and inode < n_nodes_check:
                    isforward = 1
                    new_corr = new_correlation(cc, i0, i1, inode, n_nodes, 1, BBt, edge_min, edge_max)
                    if seg_length > 0:
                        new_corr_backward = new_correlation(cc, i0, i1, inode, n_nodes, 0, BBt, edge_min, edge_max)
                        if new_corr < new_corr_backward:
                            isforward = 0
                            new_corr = new_corr_backward
                    corr_change = new_corr - corr_orig
                    corr_change_seg[ix] = corr_change
                    isforward_seg[ix] = isforward
            ix = corr_change_seg.argmax()
            corr_change = corr_change_seg[ix]
            if corr_change > 1e-3:
                improved = True
                break

        if not improved:
            break
        else:
            isforward = isforward_seg[ix]
            seg_length = (ix // n_test // n_nodes_check) + test_lengths[tl,0]
            i0 = ((ix // n_test) % n_nodes_check)
            inode = (ix % n_test)*n_skip - 1 + k%n_skip
            i1 = (i0 + seg_length)
            if corr_change > 1e-3:
                # move segment
                ishift = shift_inds(i0, i1, inode, isforward, n_nodes_check)
                inds = inds[ishift]
                iinds[edge_min : edge_max] = ishift + edge_min
                cc = shift_matrix(cc, iinds)
                corr_new = elementwise_mult_sum(cc, BBt)
                if verbose:
                    print(k, i0, i1, inode, corr_change, corr_new - corr_orig, corr_new)
                corr_orig = corr_new
                seg_len[k] = i1 - i0
    iinds = np.arange(0, n_nodes).astype(np.int32)
    iinds[edge_min : edge_max] = inds + edge_min
    return cc, iinds, seg_len


def travelling_salesman(cc, n_iter=400, alpha=1.0, edge_min=0, edge_max=-1, n_skip=None, greedy=False, verbose=False):
    """ matches correlation matrix cc to B@B.T basis functions """
    n_nodes = (cc.shape[0])
    if n_skip is None:
        n_skip = max(1, n_nodes // 30)
    cc = cc.astype(np.float32)

    if edge_max==-1:
        edge_max = n_nodes

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

        x = np.arange(0, 1.0, 1.0/n_nodes)
        BBt = compute_BBt(x, x)

        #BBt = np.tril(np.triu(BBt, -1), 1)
    else:
        BBt = np.ones((n_nodes, n_nodes))
        BBt = np.tril(np.triu(BBt, -1), 1)

    n_iter = np.int64(n_iter)
    if greedy:
        seg_iterator = itertools.combinations_with_replacement(np.arange(0, n_nodes), 2)
        test_segs = np.array([ts for ts in seg_iterator]).astype(np.int32)
        cc, inds, seg_len = tsp_greedy(cc, n_iter, n_nodes, BBt.astype(np.float32))
    else:
        cc, inds, seg_len = tsp_fast(cc, n_iter, n_nodes, n_skip, BBt.astype(np.float32), edge_min, edge_max, verbose)
        if n_skip > 1:
            cc, inds2, seg_len2 = tsp_fast(cc, n_iter, n_nodes, 1, BBt.astype(np.float32), edge_min, edge_max, verbose)
            inds = inds[inds2]

    return cc, inds, seg_len

@njit('(float32[:,:], float32[:,:], int64, int64, int64, float32[:,:],float32[:,:], boolean)', nogil=True, parallel=True)
def tsp_sub(cc, cc_add, n_iter, n_nodes, n_skip, BBt, BBt_add, verbose):
    inds = np.arange(0, n_nodes).astype(np.int32)
    seg_len = np.ones(n_iter)
    n_len = 2
    n_test = n_nodes//n_skip + 1
    test_lengths = np.arange(0, ((n_nodes - 1)//n_len)*n_len).reshape(-1,n_len)
    corr_orig = 0

    corr_change_seg = -np.inf * np.ones(n_len * n_nodes * n_test, np.float32)
    isforward_seg = -np.inf * np.ones(n_len *  n_nodes * n_test, np.float32)
    for k in range(n_iter):
        improved = False
        if corr_orig==0:
            corr_orig = elementwise_mult_sum(cc, BBt) + 4*elementwise_mult_sum(cc_add, BBt_add)
        for tl in range(len(test_lengths)):
            corr_change_seg[:] = -np.inf
            for ix in prange(0, n_len*n_nodes*n_test):
                seg_length = (ix // n_test // n_nodes) + test_lengths[tl,0]
                i0 = ((ix // n_test) % n_nodes)
                inode = (ix % n_test)*n_skip - 1 + k%n_skip
                i1 = (i0 + seg_length)
                if i1 < n_nodes and (inode < i0 or inode > i1+1) and inode < n_nodes:
                    isforward = 1
                    new_corr = new_correlation_sub(cc, cc_add, i0, i1, inode, n_nodes, 1, BBt, BBt_add)
                    if seg_length > 0:
                        new_corr_backward = new_correlation_sub(cc, cc_add, i0, i1, inode, n_nodes, 0, BBt, BBt_add)
                        if new_corr < new_corr_backward:
                            isforward = 0
                            new_corr = new_corr_backward
                    corr_change = new_corr - corr_orig
                    corr_change_seg[ix] = corr_change
                    isforward_seg[ix] = isforward
            ix = corr_change_seg.argmax()
            corr_change = corr_change_seg[ix]
            if corr_change > 1e-3:
                improved = True
                break

        if not improved:
            break
        else:
            isforward = isforward_seg[ix]
            seg_length = (ix // n_test // n_nodes) + test_lengths[tl,0]
            i0 = ((ix // n_test) % n_nodes)
            inode = (ix % n_test)*n_skip - 1 + k%n_skip
            i1 = (i0 + seg_length)
            if corr_change > 1e-3:
                # move segment
                ishift = shift_inds(i0, i1, inode, isforward, n_nodes)
                cc = shift_matrix(cc, ishift)
                cc_add = shift_rows(cc_add, ishift)
                corr_new = elementwise_mult_sum(cc, BBt) + 4*elementwise_mult_sum(cc_add, BBt_add)
                inds = inds[ishift]
                if verbose:
                    print(k, i0, i1, inode, corr_change, corr_new - corr_orig, corr_new)
                corr_orig = corr_new
                seg_len[k] = i1 - i0
    return cc, inds, seg_len

def matrix_matching(cc, BBt, cc_add, BBt_add, n_iter=400, n_skip=None, verbose=False):
    """ matches correlation matrix cc to BBt and cc_add to BBt_add """

    n_nodes = (cc.shape[0])
    if n_skip is None:
        n_skip = max(1, n_nodes // 30)
    cc = cc.astype(np.float32)
    cc_add = cc_add.astype(np.float32)
    BBt = BBt.astype(np.float32)
    BBt_add = BBt_add.astype(np.float32)

    n_components = 1

    n_iter = np.int64(n_iter)
    cc, inds, seg_len = tsp_sub(cc, cc_add, n_iter, n_nodes, n_skip, BBt, BBt_add, verbose)
    if n_skip > 1:
        cc, inds2, seg_len2 = tsp_sub(cc, cc_add, n_iter, n_nodes, 1, BBt, BBt_add, verbose)
        inds = inds[inds2]

    return cc, inds, seg_len

def compute_BBt(xi, yi, alpha=1e3):
    ds = np.abs(xi[:,np.newaxis] - yi)
    mask = ds < 1/len(xi) + .001
    BBt = - np.log(1e-10 + ds) * (1 + 5*mask)
    #BBt = 1 / (1 + alpha * (xi[:,np.newaxis] - yi)**2)**0.5
    return BBt

def cluster_split_and_sort(U, n_clusters=50, nc=25, n_splits=4, alpha=1.0, sticky=True):
    U_nodes, imax = kmeans(U, n_clusters=n_clusters)
    cc = U_nodes @ U_nodes.T
    cc,inds,seg_len = travelling_salesman(cc, verbose=True, alpha=alpha)
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
