import numpy as np

def bin1d(X, bin_size, axis=0):
    """ bin over axis of data with bin bin_size """
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        Xb = Xb[:size[axis]//bin_size*bin_size].reshape((size[axis]//bin_size, bin_size, -1)).mean(axis=1)
        Xb = Xb.swapaxes(axis, 0)
        size[axis] = Xb.shape[axis]
        Xb = Xb.reshape(size)
        return Xb
    else:
        return X

def split_testtrain(n_t, frac=0.25):
    ''' this returns indices of testing data and training data '''
    n_segs = int(min(20, n_t/4)) #usu want 20 segs, but might not have enough frames for that
    n_len = int(n_t/n_segs)
    ninds = np.linspace(0, n_t - n_len, n_segs).astype(int)
    itest = (ninds[:,np.newaxis] + np.arange(0,n_len * frac,1,int)).flatten()
    itrain = np.ones(n_t,np.bool)
    itrain[itest] = 0
    
    return itest, itrain