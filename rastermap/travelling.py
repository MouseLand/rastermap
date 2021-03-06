import numpy as np
from numba import njit, jit, float32, int32, vectorize, prange
import itertools, time
from mapping_GM import create_ND_basis

#@vectorize([float32(float32, float32)], nopython=True, target='parallel')
#def elementwise_mult(x, y):
#    return (x * y)


def bin1d(X, tbin):
    """ bin over first axis of data with bin tbin """
    size = list(X.shape)
    X = X[:size[0]//tbin*tbin].reshape((size[0]//tbin, tbin, -1)).mean(axis=1)
    size[0] = X.shape[0]
    return X.reshape(size)


@njit('float32 (float32[:,:], float32[:,:])', nogil=True)
def elementwise_mult_sum(x, y):
    return (x * y).sum()

@njit('(float32[:,:], int32[:])', nogil=True)
def shift_matrix(cc, ishift):
    return cc[ishift][:,ishift]

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

@njit('float32 (float32[:,:], int64, int64, int64, int64, int64, float32[:,:])', nogil=True)
def new_correlation(cc, i0, i1, inode, n_nodes, isforward, BBt):
    """ compute correlation change of moving segment i0:i1+1 to position inode
    inode=-1 is at beginning of sequence
    """
    ishift = shift_inds(i0, i1, inode, isforward, n_nodes)
    cc2 = shift_matrix(cc, ishift)
    corr_new = elementwise_mult_sum(cc2, BBt)
    return corr_new


@njit('(float32[:,:], int32, int32, int32, float32[:,:], float32)', nogil=True)
def test_segment(cc, i0, i1, n_nodes, BBt, corr_orig):
    """ test movements of segment i0 and i1 and return optimal """
    test_nodes = np.arange(-1, n_nodes, 1, np.int32)
    n_test = len(test_nodes)
    if n_test > 0:
        corr_changes = np.inf * np.ones((n_test))
        forward_min = np.zeros(n_test, np.int32)
        for ip in range(len(test_nodes)):
            inode = test_nodes[ip]
            if inode < i0 or inode > i1+1:
                for isforward in [0,1]:
                    new_dist = new_correlation(cc, i0, i1, inode, n_nodes, isforward, BBt)
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

@njit('(float32[:,:], int64, int64, int64, float32[:,:])', nogil=True, parallel=True)
def tsp_fast(cc, n_iter, n_nodes, n_skip, BBt):
    inds = np.arange(0, n_nodes)
    seg_len = np.ones(n_iter)
    n_len = 2
    n_test = n_nodes//n_skip + 1
    test_lengths = np.arange(0, ((n_nodes-1)//n_len)*n_len).reshape(-1,n_len)
    corr_orig = 0
    corr_change_seg = -np.inf * np.ones(n_len * n_nodes * n_test, np.float32)
    isforward_seg = -np.inf * np.ones(n_len *  n_nodes * n_test, np.float32)
    for k in range(n_iter):
        improved = False
        if corr_orig==0:
            corr_orig = elementwise_mult_sum(cc, BBt)
        for tl in range(len(test_lengths)):
            corr_change_seg[:] = -np.inf
            for ix in prange(0, n_len*n_nodes*n_test):
                seg_length = (ix // n_test // n_nodes) + test_lengths[tl,0]
                i0 = ((ix // n_test) % n_nodes)
                inode = (ix % n_test)*n_skip - 1 + k%n_skip
                i1 = (i0 + seg_length)
                if i1 < n_nodes and (inode < i0 or inode > i1+1) and inode < n_nodes:
                    isforward = 1
                    new_corr = new_correlation(cc, i0, i1, inode, n_nodes, 1, BBt)
                    if seg_length > 0:
                        new_corr_backward = new_correlation(cc, i0, i1, inode, n_nodes, 0, BBt)
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
                inds = inds[ishift]
                cc = shift_matrix(cc, ishift)
                corr_new = elementwise_mult_sum(cc, BBt)
                print(k, i0, i1, inode, corr_change, corr_new - corr_orig, corr_new)
                corr_orig = corr_new
                seg_len[k] = i1 - i0

    return cc, inds, seg_len

def travelling_salesman(cc, n_iter=400, alpha=1.0, n_skip=None, greedy=False):
    """ matches correlation matrix cc to B@B.T basis functions """
    n_nodes = (cc.shape[0])
    if n_skip is None:
        n_skip = max(1, n_nodes // 30)
    cc = cc.astype(np.float32)

    n_components = 1
    n_X = n_nodes
    basis = .5
    B, plaw = create_ND_basis(n_components, n_X, int(n_X*basis), flag=False)
    B = B[1:]
    plaw = plaw[1:]
    n_basis, n_nodes = B.shape
    B /= plaw[:,np.newaxis] ** (alpha/2)
    B_norm = (B**2).sum(axis=0)**0.5
    B = (B / B_norm).T
    BBt = (B @ B.T) / (B**2).sum(axis=1)

    BBt = np.ones((n_X, n_X))
    BBt = np.tril(np.triu(BBt, -1), 1)

    n_iter = np.int64(n_iter)
    if greedy:
        seg_iterator = itertools.combinations_with_replacement(np.arange(0, n_nodes), 2)
        test_segs = np.array([ts for ts in seg_iterator]).astype(np.int32)
        cc, inds, seg_len = tsp_greedy(cc, n_iter, n_nodes, BBt.astype(np.float32))
    else:
        cc, inds, seg_len = tsp_fast(cc, n_iter, n_nodes, n_skip, BBt.astype(np.float32))
        if n_skip > 1:
            cc, inds2, seg_len2 = tsp_fast(cc, n_iter, n_nodes, 1, BBt.astype(np.float32))
            inds = inds[inds2]

    return cc, inds, seg_len

def embedding(U):
    """ S is raw data (time by neurons); V is time x clusters (sorted)
    """

    from mapping import upsample
    n_nodes = U.shape[1]
    iclustup, cmax = upsample(U, 1, n_nodes, n_nodes)
    isort = iclustup[:,0].argsort()

    return isort
