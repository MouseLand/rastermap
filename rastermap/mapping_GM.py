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
        if flag:
            fxx = np.floor((np.arange(K)+1)/2).astype('int')
        else:
            fxx = np.arange(K).astype('int')
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
    eta = .25
    CC, yinit, ynodes, eta = CC, y0[:,xid], y0, eta

    yf = np.zeros((NN,dims))
    niter = 501 # 201
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


def run_gmm(data, n_X=21, lam = 0.5,  basis=0.25, upsample=True):
    n_components = 2

    device = torch.device('cuda')
    n_samples = data.shape[1]
    NN = data.shape[0]

    # make basis set
    B, plaw = create_ND_basis(n_components, n_X, int(n_X*basis), flag=True)
    n_basis = B.shape[0]
    B = B.astype('float32')
    B /= plaw[:,np.newaxis]**1.2
    B = torch.from_numpy(B).to(device)

    # initialize activity traces
    V = np.zeros((data.shape[1], 4),'float32')
    V[0,0] = 1
    V[1,1] = 1
    V[2,2] = 1
    V[3,3] = 1
    V = torch.from_numpy(V).to(device)

    X = torch.from_numpy(data.astype('float32')).to(device)

    sig = (X**2).mean().cpu().numpy() * .5
    tic = time.time()
    n_iter = int(200)

    nb = np.linspace(V.shape[1], B.shape[0], n_iter-50).astype('int32')
    nb = np.hstack((nb, B.shape[0] * np.ones(50, 'int32')))
    #nb[:] = B.shape[0]
    alph = 1.
    torch0 = torch.tensor((0,)).to(device)
    for it in range(n_iter):
        #import pdb; pdb.set_trace()
        vb = V @ B[:V.shape[1]]
        vn = (vb**2).sum(axis = 0)
        xi = X @ vb
        alph = torch.maximum(xi, torch0) / vn
        P    = - ((X**2).sum(axis=1).unsqueeze(1) + alph**2 * vn - 2 * alph * xi)/(2*sig)

        Pmax = P.max(axis=1)[0].unsqueeze(1)
        eP   = torch.exp(P - Pmax)
        eP = eP / (1e-5 + eP.sum(axis=1).unsqueeze(1))
        dLdP = alph * eP

        #V, _ = torch.solve(B[:nb[it]] @ dLdP.T @ X, lam * torch.eye(nb[it]).to(device)
        #                + B[:nb[it]] @ torch.diag((alph * dLdP).sum(axis=0)) @ B[:nb[it]].T)
        #V = V.T

        UU = (B[:nb[it]] @ dLdP.T) @ X
        Va, Vs, Vb = torch.svd(UU)
        V = Vb @ Va.T

        likelihood = (torch.log(eP.sum(axis=1)) + Pmax).mean()
        if it%100==0 or it==n_iter-1:
            print('%d \t train %0.5f \t %0.2fs'%
                (it, likelihood.item(), time.time()-tic))

    imax = torch.argmax(P, axis=1).cpu().numpy()
    inds = np.array(np.unravel_index(imax, (n_X, n_X))).T
    #import pdb; pdb.set_trace()
    ypred = (alph[np.arange(NN), imax]).unsqueeze(1) * (B.T @ V.T)[imax, :]
    ypred = ypred.cpu().numpy()

    cv = (X @ V @ B).cpu().numpy()
    vnorm = ((V @ B)**2).sum(axis=0).cpu().numpy()
    cmap = mapping.assign_neurons2(vnorm, cv)
    #cmap = dLdP.cpu().numpy()
    if upsample:
        embedding = upsample_grad(cmap, n_components, n_X)
    else:
        embedding = inds

    return embedding, inds, cmap, ypred

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
