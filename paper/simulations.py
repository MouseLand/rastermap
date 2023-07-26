"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from rastermap.svd import SVD
from rastermap.utils import bin1d
from rastermap import Rastermap
import sys, os

#sys.path.insert(0, '/github/rastermap/paper/')
import metrics

def psth_to_spks(psth, mean_fr=1e-2):
    n_neurons = psth.shape[0]
    psth -= psth.min(axis=1)[:,np.newaxis]
    psth /= psth.mean(axis=1)[:,np.newaxis]
    fr = mean_fr * np.random.exponential(1, size=(n_neurons,))
    fr = np.maximum(fr, 1e-5)
    spks = np.random.poisson(psth * fr[:,np.newaxis])
    spks = spks.astype("float32")
    return spks

def powerlaw_module(n_neurons=1000, n_time=50000, alpha=1.5):
    xi = np.random.rand(n_neurons).astype("float32")
    k = np.arange(0,n_neurons)
    B = np.cos(np.pi * xi[:,np.newaxis] * k )
    
    plaw = k**(-alpha/2)
    plaw[:5] = plaw[4]
    B_norm = (B**2).sum(axis=0)**0.5
    B = (B / B_norm[:,np.newaxis])
    B *= plaw
    B = B.astype("float32")

    V = (np.random.rand(B.shape[1]+1, n_time) < 0.001).astype("float32")
    expfilt = np.exp(-np.arange(0,200)/25)
    V = np.array([np.convolve(v, expfilt, mode="full")[100:n_time+100] for v in V]).T
    V -= V.mean(axis=0)
    V = SVD(V, n_components=B.shape[1])
    V /= (V**2).sum(axis=0)**0.5
    psth = B @ V.T
    psth = np.maximum(0, B @ V.T)
    return psth, xi

def stim_tuning_module(n_neurons=1000, n_time=50000):
    """ n_neurons must be divisible by 2, n_time divisible by 500 """
    n_stims = 500
    stim_types = 15
    stims = np.zeros((stim_types, n_stims), "bool")
    stims[np.random.randint(stim_types, size=n_stims), np.arange(0, n_stims)] = True
    stim_len = n_time // n_stims 
    transients = np.zeros((stim_types, n_time), "float32")
    transients[:,::stim_len] = stims
    stim_times = np.nonzero(transients)
    
    expfilt = np.exp(-np.arange(0,200,dtype="float32")/25)
    transients = np.array([np.convolve(o, expfilt, mode="full")[:n_time] 
                            for o in transients])
    transients /= transients.max()
    
    upsample = 100
    theta_pref = np.eye(upsample*stim_types).astype("float32")
    tuning_curves = gaussian_filter1d(theta_pref, 150, axis=0, mode="constant")
    tuning_curves = tuning_curves[::upsample]
    tuning_curves /= tuning_curves.max(axis=0)
    # np.random.rand(n_neurons)
    tpref = ((stim_types * upsample) * np.sort(np.random.rand(n_neurons))).astype("int")
    xi = (tpref) / (stim_types * upsample)
    psth = tuning_curves[:,tpref].T @ transients

    return psth, xi, stim_times

def stim_sustained_module(n_neurons=1000, n_time=50000):
    n_stims = n_time//500
    stim_len = n_time // n_stims 
    stim_types = 1
    t = 0
    transients = np.zeros((stim_types, n_time), "float32")
    while t < n_time - stim_len - 100:
        if t > 0:
            t += int(min(2000, np.random.exponential(750)))
        if t < n_time - stim_len:
            ist = np.random.randint(stim_types)
            transients[ist, t] = 1
            t += stim_len
        else:
            break

    stim_times = np.nonzero(transients)
    
    n_filt = 100
    stim_resp = np.zeros((stim_types * n_filt, n_time), "float32")
    sigma = np.stack((25*np.exp(np.arange(0, n_filt)/40), 
                      5*np.exp(np.arange(0, n_filt)/40)), axis=1)
    nt = 1400
    for i in range(n_filt):
        exps = [np.exp(-np.arange(0,nt)/sig) for sig in sigma[i]]
        expfilt =  exps[0] - exps[1]
        efilt = np.zeros((nt+100), "float32")
        efilt[i:i+nt] = expfilt
        stim_resp[i :: n_filt] = np.array([np.convolve(o, efilt, mode="full")[:n_time] 
                                        for o in transients])
                                        
    tpref = n_filt * (stim_types * np.sort(np.random.rand(n_neurons))).astype("int")#np.random.randint(stim_types, size=n_neurons)
    tpref += (np.sort(np.random.rand(n_neurons)) * n_filt)[::-1].astype("int")#np.random.randint(n_filt, size=n_neurons)
    xi = (tpref) / (stim_types * n_filt)
    psth = stim_resp[tpref]
    xi = 1 - xi
    return psth, xi, stim_times

def stim_sustained_module_old(n_neurons=1000, n_time=50000):
    n_stims = n_time//500
    stim_len = n_time // n_stims 
    stim_types = 1
    t = 0
    transients = np.zeros((stim_types, n_time), "float32")
    while t < n_time - stim_len - 100:
        if t > 0:
            t += int(min(2000, np.random.exponential(750)))
        if t < n_time - stim_len:
            ist = np.random.randint(stim_types)
            transients[ist, t] = 1
            t += stim_len
        else:
            break

    stim_times = np.nonzero(transients)
    
    n_filt = 100
    stim_resp = np.zeros((stim_types * n_filt, n_time), "float32")
    sigma = np.stack((10*np.exp(np.arange(0, n_filt)/40), 
                      5*np.exp(np.arange(0, n_filt)/40)), axis=1)
    for i in range(n_filt):
        exps = [np.exp(-np.arange(0,500)/sig) for sig in sigma[i]]
        expfilt =  exps[0] - exps[1]
        efilt = np.zeros((600), "float32")
        efilt[i:i+500] = expfilt
        stim_resp[i :: n_filt] = np.array([np.convolve(o, efilt, mode="full")[:n_time] 
                                        for o in transients])
                                        
    tpref = n_filt * (stim_types * np.sort(np.random.rand(n_neurons))).astype("int")#np.random.randint(stim_types, size=n_neurons)
    tpref += (np.sort(np.random.rand(n_neurons)) * n_filt)[::-1].astype("int")#np.random.randint(n_filt, size=n_neurons)
    xi = (tpref) / (stim_types * n_filt)
    psth = stim_resp[tpref]
    xi = 1 - xi
    return psth, xi, stim_times

def sequence_module(n_neurons=1000, n_time=50000):
    xi = np.sort(np.random.rand(n_neurons))[::-1]
    psth = np.zeros((n_neurons, n_time), "float32")
    t = np.random.randint(10, 50)
    nts = 3
    n_seq=0
    seq_times = []
    while t < n_time:
        seq_len = np.random.randint(350, 700)
        velocity = gaussian_filter1d(np.random.randn(seq_len), 30)
        velocity -= velocity.min()
        velocity /= velocity.max()
        velocity *= 50
        velocity = velocity[(xi * seq_len).astype("int")]
        #velocity = velocity[ii]
        t_seq = (seq_len * xi + velocity).astype("int")
        # add random breaks
        if np.random.rand() > 0.5:
            t_seq[xi > np.random.rand()] += np.random.randint(10,50) * seq_len // 100
        for n in range(nts):
            valid = t + t_seq + n < n_time 
            psth[np.arange(0, n_neurons)[valid], t + t_seq[valid] + n] = 1
        #spks[np.arange(0, n_neurons)[valid], t + t_seq[valid]] = 1
        seq_times.append(t + t_seq[valid])
        t += t_seq.max()
        t += np.random.randint(100, 200)
        n_seq += 1
    psth = gaussian_filter1d(psth, 9, axis=1)
    xi = 1 - xi
    return psth, xi, seq_times


def make_full_simulation(n_per_module=1000, random_state=0):
    np.random.seed(random_state)
    modules = [stim_tuning_module,
                stim_sustained_module,
            sequence_module, sequence_module,
            ]
    stim_times_all = []
    for k, module in enumerate(modules):
        psth0, xi0, stimes = module(n_neurons=n_per_module)
        stim_times_all.append(stimes)
        if k > 0:
            psth = np.concatenate((psth, psth0), axis=0)
            xi = np.hstack((xi, xi0 + k))
        else:
            psth = psth0 
            xi = xi0

    psth /= psth.mean(axis=1)[:,np.newaxis]

    # compute spont
    n_spont = 2 * n_per_module
    psth_spont, xi_spont = powerlaw_module(n_neurons=
                                            psth.shape[0] + n_spont)
    psth_spont /= psth_spont.mean(axis=1)[:,np.newaxis]

    # add shared noise (spont stats)
    psth_all = psth.copy() + psth_spont[:-n_spont]

    # concatenate with spont
    psth_spont_spec = psth_spont[-n_spont:].copy()
    psth_all = np.concatenate((psth_all, psth_spont_spec))
    xi_all = np.hstack((xi, xi_spont[-n_spont:] + len(modules)))

    # compute spks
    spks = psth_to_spks(psth_all)

    # independent noise
    spks += np.random.poisson(0.03, size=spks.shape)

    iperm = np.random.permutation(len(spks))
    spks = spks[iperm]
    xi_all = xi_all[iperm]
    
    return spks, xi_all, stim_times_all, psth, psth_spont, iperm

def make_2D_simulation(filename):
    n_neurons = 30000
    basis = 0.1
    alpha = 2.

    # 2D positions for each neuron
    np.random.seed(0)
    xi = 1 * np.random.rand(n_neurons, 2)

    # basis functions in 2D
    isort0 = np.argsort(xi[:,0])
    kx,ky = np.meshgrid(np.arange(0,31), np.arange(0,31))
    kx = kx.flatten()
    ky = ky.flatten()
    kx = kx[1:]
    ky = ky[1:]
    B = np.cos(np.pi * xi[:,[0]] * kx ) * np.cos(np.pi * xi[:,[1]] * ky ) 
    plaw = ((kx**2 + ky**2)**0.5)**(-alpha/2)
    B *= plaw
    B = B.astype(np.float32)
    sv = (B**2).sum(axis=0)**0.5
    B0 = B.copy()

    # time components
    n_time = 20000
    V = np.random.randn(n_time, B0.shape[1]).astype("float32")
    V -= V.mean(axis=0)
    V = TruncatedSVD(n_components=B0.shape[1]).fit_transform(V)
    V /= (V**2).sum(axis=0)**0.5
    B = B0 @ V.T

    noise = np.random.randn(*B.shape).astype("float32")
    spks = B.copy() + 5e-3 * noise

    np.savez(filename, 
            spks=spks, xi=xi)


def tuning_stats(X_embedding, spks, stim_times):
    n_x = X_embedding.shape[0]
    n_stims = len(np.unique(stim_times[0]))
    stimes_reps = np.zeros((0,2), "int")
    tcurves = np.zeros((n_x, n_stims))
    for istim in range(n_stims):
        tinds = np.nonzero(stim_times[0]==istim)[0]
        tindst = stim_times[1][tinds]
        tindst = tindst[np.newaxis,:] + np.arange(50)[:,np.newaxis]
        tcurves[:,istim] = X_embedding[:, tindst].mean(axis=(-2,-1))
        tinds = tinds[np.random.permutation(len(tinds))]
        tinds = tinds[:(len(tinds)//2)*2]
        st = stim_times[1][tinds].reshape(-1, 2)
        stimes_reps = np.append(stimes_reps, st, axis=0)
    sids_reps = stimes_reps[:,np.newaxis] + np.arange(50)[:,np.newaxis]
    sresp = spks[:1000, sids_reps].mean(axis=-2)
    sresp = zscore(sresp, axis=1)
    cc =(sresp[:,:,0] * sresp[:,:,1]).sum(axis=1) / sresp.shape[1]
    return tcurves, cc

def sustained_stats(X_embedding, stim_times):
    n_x = X_embedding.shape[0]
    n_stims = len(stim_times)
    stimes_reps = np.zeros((0,2), "int")
    suscurves = np.zeros((n_x, n_stims))
    tinds = stim_times[1]
    tinds = tinds[np.newaxis,:] + np.arange(300)[:,np.newaxis]
    xresp = X_embedding[:, tinds].mean(axis=-1)
    return xresp     

def sequence_stats(X_embedding, stim_times):
    n_x = X_embedding.shape[0]
    n_stims = 50 # use 50 positions
    n_np = len(stim_times[0]) // n_stims # neurons per position
    nd = (len(stim_times[0]) // n_np) * n_np
    n_seq = len(stim_times) - 1 # ignore last one (might not be full)
    seqcurves = np.zeros((n_x, n_stims))

    # loop over each sequence occurrence
    for i in range(n_seq):
        pos_times = stim_times[i][nd::-1].reshape(n_stims, n_np)
        seqcurves += X_embedding[:, pos_times].mean(axis=-1)
    seqcurves /= n_seq
    return seqcurves 

def benchmark_2D(xi, isorts):
    ### Benchmarks
    Xdist = metrics.distance_matrix(xi)
    nbrs1 = NearestNeighbors(n_neighbors=1500, metric="precomputed").fit(Xdist)
    ind1 = nbrs1.kneighbors(return_distance=False)
    xd = Xdist[np.tril_indices(Xdist.shape[0], -1)]

    inds = []
    rhos = []
    for isort in isorts:
        idx = np.zeros((len(isort),1))
        idx[isort, 0] = np.arange(len(isort))
        Zdist = metrics.distance_matrix(idx)
        nbrs1 = NearestNeighbors(n_neighbors=1500, metric="precomputed").fit(Zdist)
        inds.append(nbrs1.kneighbors(return_distance=False))
        zd = Zdist[np.tril_indices(Xdist.shape[0], -1)]
        rhos.append(spearmanr(xd[::10], zd[::10]).correlation)

    knn = [10, 50, 100, 200, 400, 800, 1500]
    knn_score = np.zeros((len(knn), len(inds)))
    intersections = np.zeros(len(knn))
    for j, ind2 in enumerate(inds):
        print(j)
        for k, kni in enumerate(knn):
            for i in range(len(xi)):
                knn_score[k,j] += len(set(ind1[i, :kni]) & set(ind2[i, :kni]))
            knn_score[k,j] /= len(ind1) * kni

    return knn_score, knn, rhos


def embedding_performance(root, save=True):
    # 6000 neurons in simulation with 6 modules
    embs_all = np.zeros((10, 5, 6000, 1))
    scores_all = np.zeros((10, 2, 6, 5))
    algos = ["rastermap", "tSNE", "UMAP", "isomap", "laplacian\neigenmaps"]

    for random_state in range(10):
        dat = np.load(os.path.join(root, "simulations", f"sim_{random_state}.npz"), allow_pickle=True)
        spks = dat["spks"]
        
        # rastermap
        model = Rastermap(n_clusters=100, 
                        n_PCs=200, 
                        locality=0.8,
                        time_lag_window=10,
                        symmetric=False,
                        grid_upsample=10,
                        time_bin=10,
                        bin_size=0).fit(spks)
        embs_all[random_state, 0] = model.embedding

        # tsne
        M = metrics.run_TSNE(model.Usv, perplexities=[30])
        embs_all[random_state, 1] = M

        # umap
        M = metrics.run_UMAP(model.Usv)
        embs_all[random_state, 2] = M
        
        # isomap
        M = metrics.run_isomap(model.Usv)
        embs_all[random_state, 3] = M

        # LLE
        # M = metrics.run_LLE(model.Usv)
        # embs.append(M)

        # LE
        M = metrics.run_LE(model.Usv)
        embs_all[random_state, 4] = M

        # benchmarks
        contamination_scores, triplet_scores = metrics.benchmarks(dat["xi_all"], 
                                                        embs_all[random_state].copy())
        print(triplet_scores)
        scores_all[random_state] = np.stack((contamination_scores, triplet_scores), 
                                            axis=0)
        
        # compute stats for example sim
        if random_state == 0:
            xi_all = dat["xi_all"]
            stim_times_all = dat["stim_times_all"]

            # superneurons and correlation matrices
            spks_norm = zscore(spks, axis=1)
            X_embs = [zscore(bin1d(spks_norm[emb[:,0].argsort()], 30, axis=0), axis=1) 
                        for emb in embs_all[random_state].copy()]
            X_embs_bin = [zscore(bin1d(X_emb, 10, axis=1), axis=1) for X_emb in X_embs]
            cc_embs = [(X_emb @ X_emb.T) / X_emb.shape[1] for X_emb in X_embs_bin]
            tshifts = np.arange(0, 11)
            nn, nt = X_embs_bin[0].shape
            cc_embs_max = []
            for X_emb in X_embs_bin: 
                cc_emb_max = -1 * np.ones((nn, nn))
                for i,tshift in enumerate(tshifts):
                    cc_emb_max = np.maximum(cc_emb_max, 
                                        (X_emb[:, tshift:] @ X_emb[:, :nt-tshift].T) / (nt-tshift))
                cc_embs_max.append(cc_emb_max)

            # tuning of rastermap superneurons
            tcurves, csig = tuning_stats(X_embs[0], spks, stim_times_all[0])
            xresp = sustained_stats(X_embs[0], stim_times_all[1])
            seqcurves0 = sequence_stats(X_embs[0], stim_times_all[2])
            seqcurves1 = sequence_stats(X_embs[0], stim_times_all[3])

            # grab intermediate rastermap steps
            X_nodes = zscore(model.X_nodes, axis=1)
            n_nodes, nt = X_nodes.shape
            time_lag_window=10
            symmetric=True
            tshifts = np.arange(-time_lag_window*symmetric, time_lag_window+1)
            cc_tdelay = np.zeros((n_nodes, n_nodes, len(tshifts)), np.float32)
            for i,tshift in enumerate(tshifts):
                if tshift < 0:
                    cc_tdelay[:,:,i] = (X_nodes[:, :nt+tshift] @ X_nodes[:, -tshift:].T) / (nt-tshift)        
                else:
                    cc_tdelay[:,:,i] = (X_nodes[:, tshift:] @ X_nodes[:, :nt-tshift].T) / (nt-tshift)
            cc_tdelay[np.arange(0, n_nodes), np.arange(0, n_nodes)] = 0
            # get matching matrix
            from rastermap.sort import compute_BBt
            x = np.arange(0, 1.0, 1.0/n_nodes)[:n_nodes]
            BBt_travel = compute_BBt(x, x, locality=1.0)
            BBt_travel = np.triu(BBt_travel, 1)
            BBt_log = compute_BBt(x, x, locality=0)
            BBt_log = np.triu(BBt_log, 1)

            # get U_nodes and U_upsampled
            U_nodes = model.U_nodes
            U_upsampled = model.U_upsampled

            if save:
                np.savez(os.path.join(root, "simulations", "sim_results.npz"), 
                    xi_all=xi_all, cc_tdelay=cc_tdelay, tshifts=tshifts, 
                    BBt_log=BBt_log, BBt_travel=BBt_travel, 
                    U_nodes=U_nodes, U_upsampled=U_upsampled, cc_embs=cc_embs, 
                    X_embs=X_embs, cc_embs_max=cc_embs_max,
                    tcurves=tcurves, csig=csig, xresp=xresp, seqcurves0=seqcurves0,
                    seqcurves1=seqcurves1)
    
    if save:
        np.savez(os.path.join(root, "simulations", "sim_performance.npz"), 
                scores_all=scores_all, 
                embs_all=embs_all)

        