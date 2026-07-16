"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from numba import njit, jit, float32, int32, boolean, int64, vectorize, prange
from numba.types import Tuple

def create_ND_basis(dims, nclust, K, flag=True):
    # recursively call this function until we fill out S
    if dims == 1:
        xs = np.arange(0, nclust)
        S = np.ones((K, nclust), "float64")
        for k in range(K):
            if flag:
                S[k, :] = np.sin(np.pi + (k + 1) % 2 * np.pi / 2 + 2 * np.pi / nclust *
                                 (xs + 0.5) * int((1 + k) / 2))
            else:
                S[k, :] = np.cos(np.pi / nclust * (xs + 0.5) * k)
        S /= np.sum(S**2, axis=1)[:, np.newaxis]**.5
        if flag:
            fxx = np.floor((np.arange(K) + 1) / 2).astype("int")
        else:
            fxx = np.arange(K).astype("int")
    else:
        S0, fy = create_ND_basis(dims - 1, nclust, K, flag)
        Kx, fx = create_ND_basis(1, nclust, K, flag)
        S = np.zeros((S0.shape[0], K, S0.shape[1], nclust), np.float64)
        fxx = np.zeros((S0.shape[0], K))
        for kx in range(K):
            for ky in range(S0.shape[0]):
                S[ky, kx, :, :] = np.outer(S0[ky, :], Kx[kx, :])
                fxx[ky, kx] = ((0 + fy[ky])**2 + (0 + fx[kx])**2)**0.5
        S = np.reshape(S, (K * S0.shape[0], nclust * S0.shape[1]))
    fxx = fxx.flatten()
    ix = np.argsort(fxx)
    S = S[ix, :]
    fxx = fxx[ix]
    return S, fxx


@njit("float32 (float32[:,:], float32[:,:])", nogil=True, cache=True)
def elementwise_mult_sum(x, y):
    return (x * y).sum()


def bbt_row_bounds(BBt):
    """ first and last nonzero column of each row of BBt

    The matching kernels weight every entry of a permuted correlation matrix by
    BBt, so any column where BBt is exactly zero contributes nothing to the sum
    and can be skipped. BBt is upper triangular in the default sorting path,
    so this halves the work; for a dense user-supplied BBt the bounds simply
    cover every column and nothing is skipped.
    """
    n_rows = BBt.shape[0]
    j0 = np.zeros(n_rows, np.int64)
    j1 = np.zeros(n_rows, np.int64)
    for i in range(n_rows):
        nz = np.flatnonzero(BBt[i] != 0)
        if len(nz) > 0:
            j0[i] = nz[0]
            j1[i] = nz[-1] + 1
    return j0, j1


@njit("float32 (float32[:,:], int64[:], float32[:,:], int64[:], int64[:])", nogil=True,
      cache=True)
def corr_permuted(cc, perm, BBt, j0, j1):
    """ sum over i,j of cc[perm[i], perm[j]] * BBt[i, j]

    Equivalent to elementwise_mult_sum(cc[perm][:, perm], BBt) but computed
    without materializing the permuted matrix, which is the dominant cost when
    scoring candidate moves.
    """
    total = np.float32(0.)
    for i in range(perm.shape[0]):
        cc_row = cc[perm[i]]
        BBt_row = BBt[i]
        for j in range(j0[i], j1[i]):
            total += cc_row[perm[j]] * BBt_row[j]
    return total


@njit("float32 (float32[:,:], int64[:], float32[:,:], int64[:], int64[:])", nogil=True,
      cache=True)
def corr_permuted_rows(cc, perm, BBt, j0, j1):
    """ sum over i,j of cc[perm[i], j] * BBt[i, j] (rows permuted, columns fixed) """
    total = np.float32(0.)
    for i in range(perm.shape[0]):
        cc_row = cc[perm[i]]
        BBt_row = BBt[i]
        for j in range(j0[i], j1[i]):
            total += cc_row[j] * BBt_row[j]
    return total


@njit("(int64[:], int64, int64, int64, int64)", nogil=True, cache=True)
def fill_shift_inds(inds, i0, i1, inode, n_nodes):
    """ write the shift_inds permutation into a preallocated buffer """
    l_seg = inode - i0
    p = 0
    if l_seg >= 0:
        for k in range(0, i0):
            inds[p] = k
            p += 1
        for k in range(i1, i1 + l_seg):
            inds[p] = k
            p += 1
        for k in range(i0, i1):
            inds[p] = k
            p += 1
        for k in range(i1 + l_seg, n_nodes):
            inds[p] = k
            p += 1
    else:
        for k in range(0, inode):
            inds[p] = k
            p += 1
        for k in range(i0, i1):
            inds[p] = k
            p += 1
        for k in range(inode, i0):
            inds[p] = k
            p += 1
        for k in range(i1, n_nodes):
            inds[p] = k
            p += 1


@njit("(int64[:], int64, int64)", nogil=True, cache=True)
def reverse_segment(inds, start, length):
    for k in range(length // 2):
        tmp = inds[start + k]
        inds[start + k] = inds[start + length - 1 - k]
        inds[start + length - 1 - k] = tmp


@njit("int64[:] (int64, int64, int64, int64)", nogil=True, cache=True)
def shift_inds(i0, i1, inode, n_nodes):
    """ shift segment from i0->i1 to position inode"""
    inds = np.empty(n_nodes, np.int64)
    fill_shift_inds(inds, i0, i1, inode, n_nodes)
    return inds


@njit("(float32[:,:], int64[:])", nogil=True, cache=True)
def shift_matrix_inds(cc, ishift):
    return cc[ishift][:, ishift]


@njit("(float32[:,:], int64, int64, int64, int64)", nogil=True, cache=True)
def shift_matrix_forward(cc, i0, i1, inode, n_nodes):
    ishift = shift_inds(i0, i1, inode, n_nodes)
    cc2 = cc[ishift][:, ishift]
    return cc2, ishift


@njit("(float32[:,:], int64, int64, int64, int64, int64)", nogil=True, cache=True)
def shift_matrix(cc, i0, i1, inode, n_nodes, ordering):
    cc2, ishift = shift_matrix_forward(cc, i0, i1, inode, n_nodes)
    ilength = i1 - i0
    if ordering == 1:
        cc2[inode:inode + ilength] = cc2[inode:inode + ilength][::-1]
        cc2[:, inode:inode + ilength] = cc2[:, inode:inode + ilength][:, ::-1]
        ishift[inode:inode + ilength] = ishift[inode:inode + ilength][::-1].copy()
    return cc2, ishift


@njit("UniTuple(float32, 2) (float32[:,:], int64, int64, int64, int64, float32[:,:], int64[:], int64[:])",
      nogil=True, cache=True)
def new_correlation(cc, i0, i1, inode, n_nodes, BBt, j0, j1):
    """ compute correlation of moving segment i0:i1 to position inode

    Reversing the rows and columns of the moved segment is the same as reversing
    that segment of the permutation, so both orderings are scored straight from
    cc without building either permuted matrix.
    """
    perm = np.empty(n_nodes, np.int64)
    fill_shift_inds(perm, i0, i1, inode, n_nodes)
    ilength = i1 - i0

    corr_new = corr_permuted(cc, perm, BBt, j0, j1)
    ordering = np.float32(0.)

    if ilength > 1:
        reverse_segment(perm, inode, ilength)
        corr_new0 = corr_permuted(cc, perm, BBt, j0, j1)
        if corr_new0 > corr_new:
            corr_new = corr_new0
            ordering = np.float32(1.)

    return corr_new, ordering


@njit("(float32[:,:], int64, int64, int64, float32[:,:], boolean, int64[:], int64[:])",
      nogil=True, parallel=True, cache=True)
def tsp_fast(cc, n_iter, n_nodes, n_skip, BBt, verbose, j0, j1):
    inds = np.arange(0, n_nodes).astype(np.int32)
    iinds = np.arange(0, n_nodes).astype(np.int32)
    seg_len = -1 * np.ones(n_iter)
    start_pos = -1 * np.ones(n_iter)
    end_pos = -1 * np.ones(n_iter)
    flipped = -1 * np.ones(n_iter)
    n_len = 2
    #print(n_len, n_skip)
    n_test = n_nodes // n_skip
    test_lengths = np.arange(0, ((n_nodes) // n_len) * n_len).reshape(-1, n_len) + 1
    corr_orig = 0

    corr_change_seg = -np.inf * np.ones(n_len * n_nodes * n_test, np.float32)
    ordering_seg = -np.inf * np.ones(n_len * n_nodes * n_test, np.float32)
    for k in range(n_iter):
        improved = False
        if corr_orig == 0:
            corr_orig = elementwise_mult_sum(cc, BBt)
        for tl in range(len(test_lengths)):
            corr_change_seg[:] = -np.inf
            for ix in prange(0, n_len * n_nodes * n_test):
                ilength = (ix // n_test // n_nodes) + test_lengths[tl, 0]
                i0 = ((ix // n_test) % n_nodes)
                inode = (ix % n_test) * n_skip + k % n_skip
                i1 = (i0 + ilength)
                if i1 <= n_nodes and inode + ilength <= n_nodes:
                    corr_val, ordering_val = new_correlation(cc, i0, i1, inode, n_nodes,
                                                             BBt, j0, j1)
                    corr_change = corr_val - corr_orig
                    corr_change_seg[ix] = corr_change
                    ordering_seg[ix] = ordering_val
            #corr_change_seg += np.random.randn(corr_change_seg.shape)*(corr_change_seg**2).mean()
            ix = corr_change_seg.argmax()
            corr_change = corr_change_seg[ix]
            if corr_change > 1e-3:
                improved = True
                break

        if not improved:
            break
        else:
            ordering = ordering_seg[ix]
            ilength = (ix // n_test // n_nodes) + test_lengths[tl, 0]
            i0 = ((ix // n_test) % n_nodes)
            inode = (ix % n_test) * n_skip + k % n_skip
            i1 = (i0 + ilength)
            if corr_change > 1e-3:
                # move segment
                cc, ishift = shift_matrix(cc, i0, i1, inode, n_nodes, ordering)
                inds = inds[ishift]
                corr_new = elementwise_mult_sum(cc, BBt)
                if verbose:
                    print(k, i0, i1, inode, ordering, corr_change, corr_new - corr_orig,
                          corr_new)
                corr_orig = corr_new
                seg_len[k] = i1 - i0
                start_pos[k] = i0
                end_pos[k] = inode
                flipped[k] = ordering
    return cc, inds, seg_len, start_pos, end_pos, flipped


def traveling_salesman(cc, n_iter=400, locality=0.0, circular=False,
                         n_skip=None, BBt=None, verbose=False):
    """ matches correlation matrix cc to B@B.T basis functions """
    n_nodes = (cc.shape[0])
    if n_skip is None:
        n_skip = max(1, n_nodes // 30)
    cc = cc.astype(np.float32)

    n_components = 1

    x = np.arange(0, 1.0, 1.0 / n_nodes)[:n_nodes]
    if BBt is None:
        BBt = compute_BBt(x, x, locality=locality, circular=circular)
        if np.isinf(locality):
            BBt = np.ones((n_nodes, n_nodes))
            BBt = np.tril(np.triu(BBt, -1), 1)
        BBt = np.triu(BBt)

    n_iter = np.int64(n_iter)

    BBt32 = BBt.astype(np.float32)
    j0, j1 = bbt_row_bounds(BBt32)

    cc, inds, seg_len, start_pos, end_pos, flipped = tsp_fast(
        cc, n_iter, n_nodes, n_skip, BBt32, verbose, j0, j1)
    iter_completed = np.nonzero(seg_len == -1)[0][0]
    seg_len = seg_len[:iter_completed]
    start_pos = start_pos[:iter_completed]
    end_pos = end_pos[:iter_completed]
    flipped = flipped[:iter_completed]
    if n_skip > 1:
        cc, inds2, seg_len2, start_pos2, end_pos2, flipped2 = tsp_fast(
            cc, n_iter, n_nodes, 1, BBt32, verbose, j0, j1)
        iter_completed = np.nonzero(seg_len2 == -1)[0][0]
        seg_len = np.append(seg_len, seg_len2[:iter_completed])
        start_pos = np.append(start_pos, start_pos2[:iter_completed])
        end_pos = np.append(end_pos, end_pos2[:iter_completed])
        flipped = np.append(flipped, flipped2[:iter_completed])
        inds = inds[inds2]

    return cc, inds, BBt, seg_len, start_pos, end_pos, flipped


@njit("int64[:] (int64, int64, int64, int64, int64)", nogil=True, cache=True)
def shift_inds_sub(i0, i1, inode, isforward, n_nodes):
    """ shift segment from i0->i1 to position inode"""
    n_seg = i1 - i0 + 1
    inds = np.arange(0, n_nodes, 1, np.int64)
    inds0 = inds.copy()
    if inode > i0:
        inds[i0:i0 + inode - i1] = inds0[i1 + 1:inode + 1]
        seg_pos = inode - n_seg
    else:
        inds[inode + 1 + n_seg:i0 + n_seg] = inds0[inode + 1:i0]
        seg_pos = inode
    if isforward:
        inds[seg_pos + 1:seg_pos + 1 + n_seg] = inds0[i0:i1 + 1]
    else:
        inds[seg_pos + 1:seg_pos + 1 + n_seg] = inds0[i0:i1 + 1][::-1]
    return inds


@njit("(float32[:,:], int64[:])", nogil=True, cache=True)
def shift_matrix_sub(cc, ishift):
    return cc[ishift][:, ishift]


@njit("(float32[:,:], int64[:])", nogil=True, cache=True)
def shift_rows(cc, ishift):
    return cc[ishift]

w_add = 4

@njit(
    "float32 (float32[:,:], float32[:,:], int64, int64, int64, int64, int64, float32[:,:], float32[:,:], int64[:], int64[:], int64[:], int64[:])",
    nogil=True, cache=True)
def new_correlation_sub(cc, cc_add, i0, i1, inode, n_nodes, isforward, BBt, BBt_add, j0,
                        j1, j0_add, j1_add):
    """ compute correlation change of moving segment i0:i1+1 to position inode
    inode=-1 is at beginning of sequence
    """
    ishift = shift_inds_sub(i0, i1, inode, isforward, n_nodes)
    corr_new = (corr_permuted(cc, ishift, BBt, j0, j1) +
                w_add * corr_permuted_rows(cc_add, ishift, BBt_add, j0_add, j1_add))
    return corr_new


@njit(
    "(float32[:,:], float32[:,:], int64, int64, int64, float32[:,:],float32[:,:], boolean, int64[:], int64[:], int64[:], int64[:])",
    nogil=True, parallel=True, cache=True)
def tsp_sub(cc, cc_add, n_iter, n_nodes, n_skip, BBt, BBt_add, verbose, j0, j1, j0_add,
            j1_add):
    inds = np.arange(0, n_nodes).astype(np.int64)
    seg_len = np.ones(n_iter)
    n_len = 2
    n_test = n_nodes // n_skip + 1
    test_lengths = np.arange(0, ((n_nodes - 1) // n_len) * n_len).reshape(-1, n_len)
    corr_orig = 0

    corr_change_seg = -np.inf * np.ones(n_len * n_nodes * n_test, np.float32)
    isforward_seg = -np.inf * np.ones(n_len * n_nodes * n_test, np.float32)
    for k in range(n_iter):
        improved = False
        if corr_orig == 0:
            corr_orig = elementwise_mult_sum(
                cc, BBt) + w_add * elementwise_mult_sum(cc_add, BBt_add)
        for tl in range(len(test_lengths)):
            corr_change_seg[:] = -np.inf
            for ix in prange(0, n_len * n_nodes * n_test):
                seg_length = (ix // n_test // n_nodes) + test_lengths[tl, 0]
                i0 = ((ix // n_test) % n_nodes)
                inode = (ix % n_test) * n_skip - 1 + k % n_skip
                i1 = (i0 + seg_length)
                if i1 < n_nodes and (inode < i0 or inode > i1 + 1) and inode < n_nodes:
                    isforward = 1
                    new_corr = new_correlation_sub(cc, cc_add, i0, i1, inode, n_nodes,
                                                   1, BBt, BBt_add, j0, j1, j0_add,
                                                   j1_add)
                    if seg_length > 0:
                        new_corr_backward = new_correlation_sub(
                            cc, cc_add, i0, i1, inode, n_nodes, 0, BBt, BBt_add, j0, j1,
                            j0_add, j1_add)
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
            seg_length = (ix // n_test // n_nodes) + test_lengths[tl, 0]
            i0 = ((ix // n_test) % n_nodes)
            inode = (ix % n_test) * n_skip - 1 + k % n_skip
            i1 = (i0 + seg_length)
            if corr_change > 1e-3:
                # move segment
                ishift = shift_inds_sub(i0, i1, inode, isforward, n_nodes)
                cc = shift_matrix_sub(cc, ishift)
                cc_add = shift_rows(cc_add, ishift)
                corr_new = elementwise_mult_sum(
                    cc, BBt) + w_add * elementwise_mult_sum(cc_add, BBt_add)
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

    n_iter = np.int64(n_iter)
    j0, j1 = bbt_row_bounds(BBt)
    j0_add, j1_add = bbt_row_bounds(BBt_add)
    cc, inds, seg_len = tsp_sub(cc, cc_add, n_iter, n_nodes, n_skip, BBt, BBt_add,
                                verbose, j0, j1, j0_add, j1_add)
    if n_skip > 1:
        cc, inds2, seg_len2 = tsp_sub(cc, cc_add, n_iter, n_nodes, 1, BBt, BBt_add,
                                      verbose, j0, j1, j0_add, j1_add)
        inds = inds[inds2]

    return cc, inds, seg_len


def compute_BBt_mask(xi, yi):
    sigma = 0.5 * (xi[1] - xi[0])
    gaussian = np.exp(-(xi[:, np.newaxis] - yi)**2 / (2 * sigma**2))
    gaussian[(xi[:, np.newaxis] - yi) == 0] = 0
    return gaussian

def compute_BBt(xi, yi, locality=0, circular=False, alpha=1):
    """ can change alpha but only if not splitting """
    #if locality > 0:
    #    BBt0 = compute_BBt(xi, yi, locality=0)
    #    BBt_norm = BBt0.sum()
    #    BBt_mask_norm = compute_BBt_mask(xi, yi).sum()
    eps = 1e-3
    
    if not circular:
        if alpha==1:
            ds = np.abs(xi[:, np.newaxis] - yi)
            ds[ds == 0] = 1 - eps
            BBt = -np.log(eps + ds)
        else:
            n_nodes = len(xi)
            B, plaw = create_ND_basis(1, n_nodes, 5 * n_nodes, flag=False)
            B = B[1:]
            plaw = plaw[1:]
            B /= plaw[:,np.newaxis] ** (alpha/2)
            B_norm = (B**2).sum(axis=0)**0.5
            B = B / B_norm
            BBt0 = B.T @ B
    else:
        ds = np.abs(xi[len(xi)//2] - yi)
        ds[ds == 0] = 1 - eps
        BBt0 = -np.log(eps + ds)
        BBt = np.zeros((len(xi), len(yi)), "float32")
        for k in range(len(xi)):
            BBt[k] = np.roll(BBt0, -len(xi)//2 + k)
    
    BBt_norm = BBt.mean()
    BBt /= BBt_norm
        
    if locality > 0:
        BBt_mask = compute_BBt_mask(xi, yi)
        # need to make BBt and BBt_mask on same scale
        BBt_mask_norm = BBt_mask.mean()
        BBt_mask /= BBt_mask_norm
        BBt = BBt * (1 - locality) + BBt_mask * locality
        #BBt *= BBt_norm
    
    return BBt
