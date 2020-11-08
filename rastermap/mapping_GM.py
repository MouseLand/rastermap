import time
import torch
from torch import optim
import numpy as np
from mapping import svdecon
import mapping
from matplotlib import pyplot as plt
import cv2

def preprocess(filename):
    data = np.load(filename)
    u, s, v = svdecon(data, k = 1000)
    data = u * s**.5
    return data

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
        fxx = np.floor((np.arange(K)+1)/2).astype('int')
        #fxx = np.arange(K).astype('int')
    else:
        S0, fy = create_ND_basis(dims-1, nclust, K, flag)
        Kx, fx = create_ND_basis(1, nclust, K, flag)
        S = np.zeros((S0.shape[0], K, S0.shape[1], nclust), np.float64)
        fxx = np.zeros((S0.shape[0], K))
        print(S0.shape)
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
    if dims==1:
        return S, fxx
    else:
        return S[1:], fxx[1:]

def upsample_grad(CC, dims, nX):
    NN, nnodes = CC.shape
    CC /= np.amax(CC, axis=1)[:, np.newaxis]
    xid = np.argmax(CC, axis=1)

    ys, xs = np.meshgrid(np.arange(nX), np.arange(nX))
    y0 = np.vstack((xs.flatten(), ys.flatten()))
    eta = .1
    CC, yinit, ynodes, eta = CC, y0[:,xid], y0, eta
    
    yf = np.zeros((NN,dims))
    niter = 201 # 201
    eta = np.linspace(eta, eta/10, niter)
    sig = 1.
    alpha = 1.
    y = yinit.T.copy()
    
    device = torch.device('cuda')
    CC = torch.from_numpy(CC.astype('float32')).to(device)
    y = torch.from_numpy(y.astype('float32')).to(device)
    ynodes = torch.from_numpy(ynodes.astype('float32')).to(device)
    for it in range(niter):
        yy0 =  y.unsqueeze(2) - ynodes
        K = torch.exp(-(yy0**2).sum(axis=1)/2)#(2*sig**2))
        x = (K * CC).sum(axis=1, keepdim=True) / (K**2).sum(axis=1, keepdim=True)
        err = x * K - CC
        Kprime = - x.unsqueeze(1) * yy0 * K.unsqueeze(1)
        dy = (Kprime * err.unsqueeze(1)).sum(axis=-1)
        y = y - eta[it] * dy
    
    return y.cpu().numpy()

def run_gmm(data, n_X=21, basis=0.25, lam=1, upsample=True):
    n_components = 2

    device = torch.device('cuda')
    n_samples = data.shape[1]
    init_sort = np.argsort(data[:,:2],axis=0).astype(np.float64)
    NN = data.shape[0]
    xid = np.zeros(NN, 'int')
    for j in range(2):
        iclust = np.floor(n_X * init_sort[:,j]/NN)
        xid = n_X * xid + iclust
    V = np.zeros((n_samples, n_X**2))
    for j in range(n_X**2):
        ix = xid==j
        if ix.sum():
            V[:,j] = data[xid==j].sum(axis=0)
    B, plaw = mapping.create_ND_basis(n_components, n_X, int(n_X*basis), flag=True)
    plaw[0] = 1000
    n_basis = B.shape[0]
    B = B.astype('float32')
    V = V @ B.T
    V = V.astype('float32')
    B /= np.maximum(1 , plaw[:,np.newaxis])
    V0 = V / (V**2).sum(axis=0)**0.5
    B = torch.from_numpy(B).to(device)

    V = V0.copy()
    #V = .1*np.eye(1000,25).astype('float32') #np.zeros(V.shape, 'float32')
    V = np.zeros(V.shape,'float32')
    V[0,1:3] = [1, -1]
    V[1,3:5] = [1, -1]
    V = .01 * V
    X = torch.from_numpy(data[:,:,np.newaxis].astype('float32')).to(device)
    #X -= X.mean(axis=0)
    V = torch.from_numpy(V).to(device)
    sig = (X**2).mean() * .25
    tic = time.time()
    n_iter = int(500)
    dV = torch.zeros_like(V)
    LAM = np.linspace(lam, lam, n_iter-20)
    for it in range(n_iter+1):
        #if it==n_iter:
        #    lam = 0
        if it<len(LAM):
            lam = LAM[it]
        P    = - ((X**2).sum(axis=1) + ((V @ B)**2).sum(axis=0) - 2 * X[:,:,0]@V@B)/(2*sig)
        Pmax = P.max(axis=1)[0].unsqueeze(1)
        eP   = torch.exp(P - Pmax)
        dLdP = eP / eP.sum(axis=1).unsqueeze(1)
        V, _ = torch.solve(B @ dLdP.T @ X[:,:,0], lam * torch.eye(B.shape[0]).to(device) 
                        + B @ torch.diag(dLdP.sum(axis=0)) @ B.T)
        V = V.T
        likelihood = (torch.log(torch.exp(P - Pmax).sum(axis=1)) + Pmax).mean()
        if it%100==0 or it==n_iter:
            print('%d \t train %0.5f \t %0.2fs'%
                (it, likelihood.item(), time.time()-tic))
    inds = np.array(np.unravel_index(torch.argmax(P, axis=1).cpu().numpy(), (n_X, n_X))).T
    
    if upsample:
        cv = (X[:,:,0] @ V @ B).cpu().numpy()
        vnorm = ((V @ B)**2).sum(axis=0).cpu().numpy()
        cmap = mapping.assign_neurons2(vnorm, cv)
        embedding = upsample_grad(np.sqrt(cmap), n_components, n_X)
    else:
        embedding = inds

    return embedding, inds, cmap

def ziegler_colors():
    cmap = cv2.imread('figs/ziegler.png')
    cmap = cmap.astype(float)/255
    cz = cv2.resize(cmap, (41,41))
    cz = cz.reshape((-1,3))
    return cz

def plot_output(embedding, colors=None, rand=0, alpha=.25):
    NN = embedding.shape[0]
    #embedding = np.array(np.unravel_index(xid.astype('int'), (n_X, n_X))).T
    plt.figure(figsize=(8,8))
    if colors is None:
        plt.scatter(embedding[:,0]+np.random.randn(NN)*rand, 
                    embedding[:,1]+np.random.randn(NN)*rand, s=8, alpha = alpha)
    else:
        plt.scatter(embedding[:,0]+np.random.randn(NN)*rand, 
                    embedding[:,1]+np.random.randn(NN)*rand, s=8, alpha = alpha, c=colors, cmap='jet')
        
    plt.show()