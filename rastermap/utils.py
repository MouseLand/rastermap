import numpy as np

def bin1d(X, tbin):
    """ bin over first axis of data with bin tbin """
    size = list(X.shape)
    X = X[:size[0]//tbin*tbin].reshape((size[0]//tbin, tbin, -1)).mean(axis=1)
    size[0] = X.shape[0]
    return X.reshape(size)

def split_testtrain(n_t, frac=0.25):
    ''' this returns indices of testing data and training data '''
    n_segs = int(min(20, n_t/4)) #usu want 20 segs, but might not have enough frames for that
    n_len = int(n_t/n_segs)
    ninds = np.linspace(0, n_t - n_len, n_segs).astype(int)
    itest = (ninds[:,np.newaxis] + np.arange(0,n_len * frac,1,int)).flatten()
    itrain = np.ones(n_t,np.bool)
    itrain[itest] = 0
    
    return itest, itrain