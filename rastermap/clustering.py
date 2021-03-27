import numpy as np
from numba import njit, jit, float32, int32, boolean, int64, vectorize, prange
from numba.types import Tuple
import itertools, time
from sklearn.cluster import KMeans
from scipy.stats import zscore

def kmeans(X, n_clusters=100):
    model = KMeans(n_init=1, n_clusters=n_clusters, random_state=0).fit(X)
    X_nodes = model.cluster_centers_ 
    X_nodes = X_nodes / (1e-10 + ((X_nodes**2).sum(axis=1))[:,np.newaxis])**.5  
    imax = model.labels_

    #cc = X @ X_nodes.T 
    #imax = cc.argmax(axis=1)

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

@njit('(float32[:,:], int64[:])', nogil=True)
def shift_matrix_inds(cc, ishift):
    return cc[ishift][:,ishift]

@njit('(float32[:,:], int64, int64, int64, int64)', nogil=True)
def shift_matrix_forward(cc, i0, i1, inode, n_nodes):
    ishift = shift_inds(i0, i1, inode, n_nodes)
    cc2 = cc[ishift][:,ishift]
    return cc2, ishift

@njit('(float32[:,:], int64, int64, int64, int64, int64)', nogil=True)
def shift_matrix(cc, i0, i1, inode, n_nodes, ordering):
    cc2, ishift = shift_matrix_forward(cc, i0, i1, inode, n_nodes)
    ilength = i1 - i0
    if ordering==1:
        cc2[inode:inode+ilength] = cc2[inode:inode+ilength][::-1]
        cc2[:, inode:inode+ilength] = cc2[:, inode:inode+ilength][:, ::-1]
        ishift[inode:inode+ilength] = ishift[inode:inode+ilength][::-1]
    return cc2, ishift

@njit('float32[:] (float32[:,:], int64, int64, int64, int64, float32[:,:])', nogil=True)
def new_correlation(cc, i0, i1, inode, n_nodes, BBt):
    """ compute correlation change of moving segment i0:i1+1 to position inode
    inode=-1 is at beginning of sequence
    """
    cc_new = shift_matrix_forward(cc, i0, i1, inode, n_nodes)[0]
    ilength = i1 - i0
    
    corr_new = elementwise_mult_sum(cc_new, BBt)
    ordering = 0

    if ilength > 1:
        cc0 = cc_new.copy()
        cc0[inode:inode+ilength] = cc0[inode:inode+ilength][::-1]
        cc0[:, inode:inode+ilength] = cc0[:, inode:inode+ilength][:, ::-1]

        corr_new0 = elementwise_mult_sum(cc0, BBt)
        if corr_new0 > corr_new:
            corr_new = corr_new0 
            ordering = 1

    corr_out = np.zeros(2, np.float32)
    corr_out[0] = corr_new
    corr_out[1] = ordering

    return corr_out

@njit('(float32[:,:], int64, int64, int64, float32[:,:], boolean)', nogil=True, parallel=True)
def tsp_fast(cc, n_iter, n_nodes, n_skip, BBt, verbose):
    inds = np.arange(0, n_nodes).astype(np.int32)
    iinds = np.arange(0, n_nodes).astype(np.int32)
    seg_len = np.ones(n_iter)
    n_len = 2
    n_test = n_nodes//n_skip
    test_lengths = np.arange(0, ((n_nodes)//n_len)*n_len).reshape(-1,n_len) + 1
    corr_orig = 0

    corr_change_seg = -np.inf * np.ones(n_len * n_nodes * n_test, np.float32)
    ordering_seg = -np.inf * np.ones(n_len *  n_nodes * n_test, np.float32)
    for k in range(n_iter):
        improved = False
        if corr_orig==0:
            corr_orig = elementwise_mult_sum(cc, BBt) 
        for tl in range(len(test_lengths)):
            corr_change_seg[:] = -np.inf
            for ix in prange(0, n_len*n_nodes*n_test):
                ilength = (ix // n_test // n_nodes) + test_lengths[tl,0]
                i0 = ((ix // n_test) % n_nodes)
                inode = (ix % n_test)*n_skip + k%n_skip
                i1 = (i0 + ilength)
                if i1 <= n_nodes and inode + ilength <= n_nodes:
                    new_corr = new_correlation(cc, i0, i1, inode, n_nodes, BBt)
                    corr_change = new_corr[0] - corr_orig
                    corr_change_seg[ix] = corr_change
                    ordering_seg[ix] = new_corr[1]
            ix = corr_change_seg.argmax()
            corr_change = corr_change_seg[ix]
            if corr_change > 1e-3:
                improved = True
                break

        if not improved:
            break
        else:
            ordering = ordering_seg[ix]
            ilength = (ix // n_test // n_nodes) + test_lengths[tl,0]
            i0 = ((ix // n_test) % n_nodes)
            inode = (ix % n_test)*n_skip + k%n_skip
            i1 = (i0 + ilength)
            if corr_change > 1e-3:
                # move segment
                cc, ishift = shift_matrix(cc, i0, i1, inode, n_nodes, ordering)
                inds = inds[ishift]
                corr_new = elementwise_mult_sum(cc, BBt)
                if verbose:
                    print(k, i0, i1, inode, ordering, corr_change, corr_new - corr_orig, corr_new)
                corr_orig = corr_new
                seg_len[k] = i1 - i0
    return cc, inds, seg_len


def travelling_salesman(cc, n_iter=400, alpha=1.0, n_skip=None, verbose=False):
    """ matches correlation matrix cc to B@B.T basis functions """
    n_nodes = (cc.shape[0])
    if n_skip is None:
        n_skip = max(1, n_nodes // 30)
    cc = cc.astype(np.float32)

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

        #x = np.arange(0, 1.0, 1.0/n_nodes)[:n_nodes]
        #BBt = compute_BBt(x, x)
    else:
        BBt = np.ones((n_nodes, n_nodes))
        BBt = np.tril(np.triu(BBt, -1), 1)

    n_iter = np.int64(n_iter)
    
    cc, inds, seg_len = tsp_fast(cc, n_iter, n_nodes, n_skip, BBt.astype(np.float32), verbose)
    if n_skip > 1:
        cc, inds2, seg_len2 = tsp_fast(cc, n_iter, n_nodes, 1, BBt.astype(np.float32), verbose)
        inds = inds[inds2]

    return cc, inds, seg_len

@njit('int32[:] (int64, int64, int64, int64, int64)', nogil=True)
def shift_inds_sub(i0, i1, inode, isforward, n_nodes):
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
def shift_matrix_sub(cc, ishift):
    return cc[ishift][:,ishift]

@njit('(float32[:,:], int32[:])', nogil=True)
def shift_rows(cc, ishift):
    return cc[ishift]

@njit('float32 (float32[:,:], float32[:,:], int64, int64, int64, int64, int64, float32[:,:], float32[:,:])', nogil=True)
def new_correlation_sub(cc, cc_add, i0, i1, inode, n_nodes, isforward, BBt, BBt_add):
    """ compute correlation change of moving segment i0:i1+1 to position inode
    inode=-1 is at beginning of sequence
    """
    ishift = shift_inds_sub(i0, i1, inode, isforward, n_nodes)
    cc2 = shift_matrix_sub(cc, ishift)
    cc_add2 = shift_rows(cc_add, ishift)
    corr_new = elementwise_mult_sum(cc2, BBt) + 4*elementwise_mult_sum(cc_add2, BBt_add)
    return corr_new


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
                ishift = shift_inds_sub(i0, i1, inode, isforward, n_nodes)
                cc = shift_matrix_sub(cc, ishift)
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
    BBt = - np.log(1e-10 + ((xi[:,np.newaxis] - yi)**2)**0.5)
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

############################################
#### MAKE MANY CLUSTERS, PAIR THEN SORT ####
############################################

@njit('(int64, int64)', nogil=True)
def ix_to_pos(ix, n_pairs):
    i0 = ix // n_pairs
    i1 = ix % n_pairs
    i1 = i1 + 1 if i1==i0 else i1
    return i0, i1

@njit('(int64[:,:], int64, int64)', nogil=True)
def pos_to_breaks(pairs, i0, i1):
    inds = np.nonzero(pairs==i0)
    ipos0 = inds[0][0]
    break0 = pairs[inds[0][0], (inds[1][0]+1)%2]
    inds = np.nonzero(pairs==i1)
    ipos1 = inds[0][0]
    break1 = pairs[inds[0][0], (inds[1][0]+1)%2]
    return ipos0, break0, ipos1, break1

@njit('(float32[:,:], int64)', nogil=True, parallel=True)  
def bunch_clusters(cc, verbose):
    """ n_clusters MUST be divisible by 2 """
    n_clusters = cc.shape[0]
    np.random.seed(0)
    pairs = np.random.permutation(n_clusters).reshape(-1, 2)
    pairs_orig = pairs.copy()
    n_pairs = pairs.shape[0]
    corr_change_seg = -np.inf * np.ones(n_pairs * (n_pairs-1))
    perm_seg = np.zeros(n_pairs * (n_pairs-1), np.int64)
    n_iter = max(100, 2 * n_clusters)
    corr_orig = np.diag(cc[pairs[:,0]][:,pairs[:,1]]).sum()
    n_perms = 2

    for k in range(n_iter):
        corr_change_seg[:] = -np.inf
        for ix in prange(n_pairs * (n_pairs-1)):
            i0, i1 = ix_to_pos(ix, n_pairs)
            corr_new0 = corr_orig 
            corr_new0 -= cc[pairs[i0,0], pairs[i0,1]] + cc[pairs[i1,0], pairs[i1,1]]
            corr_new = corr_new0
            for i in range(n_perms):
                corr_new1 = corr_new0 + cc[pairs[i0,0], pairs[i1,i]] + cc[pairs[i0,1], pairs[i1,1-i]]
                if corr_new1 > corr_new:
                    corr_new = corr_new1
                    perm_seg[ix] = i
            corr_change_seg[ix] = corr_new - corr_orig
        ix = corr_change_seg.argmax()
        corr_change = corr_change_seg[ix]
        perm = perm_seg[ix]
        if corr_change > 1e-3:
            i0, i1 = ix_to_pos(ix, n_pairs)
            iswap0 = pairs[i0,1]
            iswap1 = pairs[i1,perm]
            iswap2 = pairs[i1,1-perm]
            pairs[i0,1] = iswap1
            pairs[i1,0] = iswap0
            pairs[i1,1] = iswap2
            corr_new = np.diag(cc[pairs[:,0]][:, pairs[:,1]]).sum()
            if verbose:
                print(k, corr_new, corr_change, corr_new-corr_orig)
            corr_orig=corr_new
        else:
            break
    return pairs

def add_split_inds(inds, ind, split, i, icurrent):
    """ label each node with its hierarchical cluster identity """
    #if split==-1:     
    if split < len(inds):
        n_splits = len(inds)
        for jj, j in enumerate(np.nonzero(inds[split]==i)[0]):
            jcurrent = icurrent + jj * 2**(n_splits - split - 2)
            if split==len(inds)-1:
                ind[j] = jcurrent
            else:
                add_split_inds(inds, ind, split+1, j, jcurrent)
    else:
        return 0

def pair_clusters(U_nodes, cc, n_neurons, nc=50):
    """ pair clusters to reduce sorting problem """
    n_nodes = U_nodes.shape[0]
    n_splits = int(np.log(n_nodes // nc) / np.log(2))
    cc_pair = cc.copy()
    U_pair = U_nodes.copy()
    pair_splits = []
    U_pairs = [U_pair]
    n_neurons_pair = n_neurons
    # pair clusters
    for k in range(n_splits):
        n_nodes = cc_pair.shape[0]
        pairs = bunch_clusters(cc_pair.astype(np.float32), verbose=False)
        pair_splits.append(pairs)
        U_pair = ((U_pair[pairs] * n_neurons_pair[pairs][:,:,np.newaxis]).sum(axis=1) 
                      / n_neurons_pair[pairs].sum(axis=1)[:,np.newaxis])
        cc_pair = U_pair @ U_pair.T
        U_pairs.append(U_pair)
        n_neurons_pair = n_neurons[pairs].sum(axis=1)
        
    # get relative indices for pairs
    inds = []
    for k in range(n_splits):
        nk = pair_splits[-(k+1)].shape[0]
        inds.append(np.zeros(nk*2, np.int64))
        inds[-1][pair_splits[-(k+1)]] = np.tile(np.arange(0, nk)[:,np.newaxis],(1,2))
        
    # use relative indices to create hierarchical indices
    ind = np.zeros(nc * 2**n_splits, np.int64)
    for i in range(nc):
        icurrent = i * 2**(n_splits-1)
        add_split_inds(inds, ind, 0, i, icurrent)
        
    return ind

def sort_paired_clusters(U_nodes, ind, n_neurons, nc=50, nch=50):
    n_nodes, n_PCs = U_nodes.shape
    n_splits = int(np.log(n_nodes // nc) / np.log(2))
    ind_sort = ind.copy()
    for k in range(n_splits):
        nn = nc * 2**k # number of nodes at this level
        div = 2**(n_splits-k-1) # number of 0-level pairs at this level
        Un = np.zeros((nn, n_PCs))
        for n in range(nn):
            inn = ind_sort//div == n
            Un[n] = ((U_nodes[inn] * n_neurons[inn][:,np.newaxis]).sum(axis=0) /
                      n_neurons[inn][:,np.newaxis].sum(axis=0))
        ind_new = np.zeros(n_nodes, np.int64)
        if k==0:
            nc0 = nc
        else:
            nc0 = nch
        
        for i in range(nn//nc0):
            ii = np.arange(nn)
            node_set = np.logical_and(ii>=i*nc0, ii<(i+1)*nc0)
            U_nodes0 = Un[node_set]
            cc0 = U_nodes0 @ U_nodes0.T
            cc_add = U_nodes0 @ Un[~node_set].T
            ifrac = node_set.mean()
            x = np.linspace(i*nc0/nn, (i+1)*nc0/nn, nc0+1)[:-1]
            y = np.linspace(0, 1, nn+1)[:-1][~node_set]
            BBt = compute_BBt(x, x)
            BBt -= np.diag(np.diag(BBt))

            BBt_add = compute_BBt(x, y)

            cc_out, inds, seg_len = matrix_matching(cc0, BBt, 
                                                    cc_add, BBt_add, 
                                                    verbose=False)#cc.shape[0]-25)
            icc = inds.argsort()
            for c in range(nc0):
                ichange = ind_sort//div == i*nc0 + c
                ind_new[ichange] = (i*nc0 + icc[c])*div + ind_sort[ichange]%div
        ind_sort = ind_new
    return ind_sort.argsort()

def cluster_pair_and_sort(U, n_clusters=800, nc=50):
    """ nc = n_clusters // 2**n_splits """
    U_nodes, imax = kmeans(U.astype(np.float32), n_clusters=n_clusters)
    U_nodes = U_nodes.astype(np.float64)
    cc = U_nodes @ U_nodes.T
    n_neurons = np.bincount(imax)
    ind = pair_clusters(U_nodes, cc, n_neurons, nc=nc)
    ind_final = sort_paired_clusters(U_nodes, ind, n_neurons, nc=nc)
    U_nodes = U_nodes[ind_final]
    Y_nodes = np.arange(0, U_nodes.shape[0])[:,np.newaxis]
    return U_nodes, Y_nodes
