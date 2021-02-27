import numpy as np
from numba import njit, jit, float32, int32, vectorize, prange
import itertools
from mapping_GM import create_ND_basis

@njit('(float32[:,:], float32[:,:])', nogil=True)
def elementwise_mult_sum(x, y):
    return (x * y).sum()

@njit('(float32[:,:], int32[:])', nogil=True)
def shift_matrix(cc, ishift):
    return cc[ishift][:,ishift]

@njit('int32[:] (int32, int32, int32, int32, int32)', nogil=True)
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

@njit('float32 (float32[:,:], int32, int32, int32, int32, int32, float32[:,:])', nogil=True)
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
def tsp(cc, n_iter, n_nodes, BBt, test_segs):
    inds = np.arange(0, n_nodes)
    n_segs = len(test_segs)
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
            cc = cc[ishift][:,ishift]#np.ix_(ishift, ishift)]

            new_dists = elementwise_mult_sum(cc, BBt)
            print(k, corr_change_min, new_dists - true_dists, new_dists)
        else:
            break
    return cc, inds

def travelling_salesman(cc, n_iter=100, alpha=1.0):
    """ matches correlation matrix cc to B@B.T basis functions """
    n_nodes = np.int32(cc.shape[0])
    
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

    n_iter = np.int32(n_iter)
    seg_iterator = itertools.combinations_with_replacement(np.arange(0, n_nodes), 2)
    test_segs = np.array([ts for ts in seg_iterator]).astype(np.int32)

    cc, inds = tsp(cc, n_iter, n_nodes, BBt.astype(np.float32), test_segs)

    return cc, inds
