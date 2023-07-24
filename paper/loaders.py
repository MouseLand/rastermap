"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import h5py
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import uniform_filter
from scipy.interpolate import interp1d
from scipy.sparse import csr_array 
from scipy.stats import zscore 
import mat73 
from sklearn.decomposition import TruncatedSVD
import cv2 

def proc_metadata(corridor):
    # get events for neural activity plot
    whiteSpcFrameInd = corridor['whiteSpcStart']
    run = corridor['subRun']
    Image1RewInd = corridor['Image1Rew']
    sound_inds = corridor['SoundInd']
    try:
        lick_inds = corridor['Lick']
    except:
        lick_inds = corridor['LickInd']
    VRpos = corridor['VRpos']
    WallType1 = corridor['WallType1']
    WallType2 = corridor['WallType2']

    # get corridor start and end times
    corridor_starts = np.concatenate((np.stack((WallType1, np.zeros(len(WallType1))), axis=1), 
                            np.stack((WallType2, np.ones(len(WallType2))), axis=1)), axis=0)
    corridor_starts = corridor_starts[corridor_starts[:,0].argsort()]
    corridor_starts = corridor_starts[:-1]
    corridor_widths = whiteSpcFrameInd[:-1] - corridor_starts[:,0]

    # get rewards (first lick after reward delivery)
    reward_inds = []
    for n in range(len(Image1RewInd)):
        start = Image1RewInd[n]
        start += (lick_inds[lick_inds > start] - start).min()
        reward_inds.append(start)
    reward_inds = np.array(reward_inds)

    # get images
    wallimg1 = corridor['WallType1_img']
    wallimg2 = corridor['WallType2_img']
    Ly, Lx = wallimg1.shape
    wallimg1 = cv2.resize(wallimg1, (Lx//10, Ly//10), cv2.INTER_CUBIC)
    wallimg2 = cv2.resize(wallimg2, (Lx//10, Ly//10), cv2.INTER_CUBIC)
    corridor_imgs = np.stack((wallimg1, wallimg2), axis=0)

def tuning_curves_VR(sn, VRpos, corridor_starts):
    corr_tot = np.zeros(2)
    npts=100
    n_sn = sn.shape[0]
    corridor_tuning = np.zeros((2, n_sn, npts), np.float32)
    tcorridor = np.zeros(len(VRpos), "bool")
    for n in range(len(corridor_starts)-1):
        start = int(np.round(corridor_starts[n,0]))
        end = int(np.round(corridor_starts[n+1,0])) #int(np.ceil(whiteSpcFrameInd[n]))
        icorr = int(corridor_starts[n,1])
        f = interp1d(VRpos[start:end], sn[:,start:end], 
                    bounds_error=False, 
                    fill_value=(sn[:,start], sn[:,end]))
        corridor_tuning[icorr,:,:] += f(np.linspace(0, 1, npts))
        tcorridor[start:end] = icorr
        corr_tot[icorr] += 1
    corridor_tuning /= corr_tot[:,np.newaxis,np.newaxis]
    return corridor_tuning

def rolling_max(x):
    idx_split = [np.nonzero(~np.isnan(x))[0][0]];
    
    for idx, val in enumerate(x):
        if val < 0.025 or val > 1.575:
            tmp = x[max(idx-20,0):min(idx+20,len(x))];
            if val == min(tmp) or val==max(tmp):
                idx_split.append(idx)
    idx_split = np.array(idx_split)
    return idx_split


def tuning_curves_hipp(spks, locs, bins):
    ibin = np.digitize(locs, bins) - 1
    n_bins = ibin.max()
    inan = np.isnan(locs)
    ibin[inan] = -1
    tcurves = np.zeros((spks.shape[0], n_bins))
    for b in range(n_bins):
        tcurves[:, b] = spks[:, ibin==b].mean(axis=1)
    return tcurves


def load_hippocampus_data(filename, bin_sec=0.2):
    """ adapted from Ding Zhou's code

    code: https://github.com/zhd96/pi-vae/blob/main/code/rat_preprocess_data.py
    paper: https://arxiv.org/abs/2011.04798

    includes inter-maze intervals and fast-spiking neurons

    """
    f = mat73.loadmat(filename)

    ## load spike info
    st = np.array(f["sessInfo"]["Spikes"]["SpikeTimes"])
    clu = np.array(f["sessInfo"]["Spikes"]["SpikeIDs"])
    pyr_cells = np.array(f["sessInfo"]["Spikes"]["PyrIDs"])
    cells, clu = np.unique(clu, return_inverse=True)
    pyr_cells = np.isin(cells, pyr_cells)
    
    ## load location info ## all in maze
    locations_2d = np.array(f["sessInfo"]["Position"]["TwoDLocation"])
    locations = np.array(f["sessInfo"]["Position"]["OneDLocation"])
    nnan = np.nonzero(~np.isnan(locations))[0]
    inan = np.nonzero(np.isnan(locations))[0]
    fi = interp1d(nnan, locations[nnan], 
                    bounds_error=False)#, kind="nearest")
    locations_raw = locations.copy()
    locations[inan] = fi(inan)
    for j in range(2):
        nnan = np.nonzero(~np.isnan(locations_2d[:,j]))[0]
        inan = np.nonzero(np.isnan(locations_2d[:,j]))[0]
        fi = interp1d(nnan, locations_2d[nnan,j], bounds_error=False)#, kind="nearest")
        locations_2d[inan,j] = fi(inan)
    locations_2d[locations_2d[:,1] > 0.1, 1] = 0 # some weirdness at the beginning of the session
    locations_times = np.array(f["sessInfo"]["Position"]["TimeStamps"]).flatten();
    
    keys = list(f["sessInfo"]["Epochs"].keys())
    epochs = []
    for key in keys:
        epochs.append(np.array(f["sessInfo"]["Epochs"][key]))

    spks = csr_array((np.ones(len(st), "uint8"), 
                    (clu, np.floor(st / bin_sec).astype("int"))))

    maze_epoch = epochs[keys.index("MazeEpoch")].flatten()
    maze_bins = np.floor(maze_epoch / bin_sec).astype("int")
    spks_maze = spks[:, maze_bins[0] : maze_bins[1]].todense().astype("float32")
    
    # locations in spks time frame
    locations_bins = np.floor(locations_times/bin_sec).astype("int") - maze_bins[0]
    locations_vec = np.nan * np.zeros(spks_maze.shape[1])
    locations_raw_vec = np.nan * np.zeros(spks_maze.shape[1])
    locations_2d_vec = np.nan * np.zeros((spks_maze.shape[1],2))
    locations_vec[locations_bins] = locations
    locations_raw_vec[locations_bins] = locations_raw
    locations_2d_vec[locations_bins] = locations_2d

    # trial splits
    splits = rolling_max(locations_vec)
    splits = np.delete(splits, np.where(np.abs(np.diff(locations_vec[splits])) < 1)[0])
    loc = locations_vec.copy()
    loc -= loc[~np.isnan(loc)].min() 
    loc /= loc[~np.isnan(loc)].max()
    loc_signed = np.nan * np.zeros(len(loc))
    for j in range(len(splits)-1):
        slc = slice(splits[j], splits[j+1])
        loc_signed[slc] = loc[slc] * np.sign(2*((j+1)%2) - 1)    

    ## return as dictionary
    dat = {"spks":spks_maze, 
            "loc":locations_vec, "loc_signed": loc_signed,
            "inan_loc": np.isnan(locations_raw_vec), 
            "loc2d":locations_2d_vec,
            "pyr_cells":pyr_cells, "splits": splits}
    return dat


def load_fish_data(root, subject):
    from suite2p import default_ops
    from suite2p.extraction import preprocess, oasis

    dat = loadmat(os.path.join(root, f"subject_{subject}", "data_full.mat"), 
                   squeeze_me=True)
    fish = dat["data"].item()
    with h5py.File(os.path.join(root, f"subject_{subject}", "TimeSeries.h5"),"r") as f:
        F = np.array(f["CellResp"]).T
        
    n_neurons = fish[7]
    igood = np.ones(n_neurons, "bool")
    igood[fish[9]] = 0

    xyz = fish[8][igood]
    n_neurons = igood.sum()
    
    ops = default_ops()
    fs = 2 # sampling rate in Hz
    tau = 2 # sensor timescale

    Fcorr = preprocess(F, ops["baseline"], ops["win_baseline"], ops["sig_baseline"], fs)
    S = oasis(Fcorr, ops["batch_size"], tau, fs)
    
    stims = fish[14]-1
    swimming = fish[19].T
    eyepos = fish[21].T
    
    del dat
    
    # take only high variance neurons (other neurons are noisy)
    good_neurons = F.std(axis=1) > 0.08
    S = S[good_neurons]
    F = F[good_neurons]
    xyz = xyz[good_neurons]
    xyz[:,0] = -1*xyz[:,0]

    return S, F, xyz, stims, swimming, eyepos

def load_widefield_data(root):
    USV = mat73.loadmat(os.path.join(root, "Vc.mat"))
    U = np.array(USV["U"])
    Ly,Lx = U.shape[:-1]
    U = U.reshape(Ly*Lx, -1)
    n_comps, Lyx = U.shape
    interpVc  = loadmat(os.path.join(root, "interpVc.mat"),squeeze_me=True)
    Vsv = np.array(interpVc["Vc"].T)
    n_time = Vsv.shape[0]
    sv = np.nansum(Vsv**2, axis=0)**0.5
    inan = np.isnan(U).sum(axis=1)>0
    # erode noisy edges of widefield (specific to data processing)
    img = np.ones((Ly * Lx))
    img[inan] = 0
    img = img.reshape(Ly, Lx)
    img = uniform_filter(img, 25)
    inan = img.flatten() < 0.99
    goodid = np.nonzero(~inan)[0]
    U0 = U[~inan]

    # xpos, ypos of each voxel
    x, y = np.meshgrid(np.arange(0, Ly), np.arange(0, Lx), indexing="ij")
    x, y = x.flatten(), y.flatten()
    xpos = x[~inan]
    ypos = y[~inan]

    # load behavior and task events
    sevents = mat73.loadmat(os.path.join(root, "regData.mat"))
    recIdx = sevents["recIdx"][~sevents["idx"]]
    recLabels = sevents["recLabels"]
    regressors = sevents["fullR"].astype("float32")

    # split regressors into behavior vs task
    is_behav = np.ones(len(recLabels), "bool")
    is_behav[:3] = False
    is_behav[9:18] = False
    behav_idx = np.isin(recIdx, np.nonzero(is_behav)[0] + 1)

    # get event times 
    reward_times = np.nonzero(regressors[:, np.nonzero(recIdx==17)[0][0]])[0]
    stim_times, lick_times, handle_times = [], [], []
    stim_labels = ["vis L", "vis R", "aud L", "aud R"]
    lick_labels = ["left lick", "right lick"]
    handle_labels = ["left handle", "right handle"]
    for i in [3,5]:
        print(recLabels[i])
        handle_times.append(np.nonzero(regressors[:, np.nonzero(recIdx==i+1)[0][0]])[0])
    for i in range(7,9):
        print(recLabels[i])
        lick_times.append(np.nonzero(regressors[:, np.nonzero(recIdx==i+1)[0][0]])[0])
    for i in range(9,13):
        print(recLabels[i])
        stim_times.append(np.nonzero(regressors[:, np.nonzero(recIdx==i+1)[0][0]])[0])

    regressors -= regressors.mean(axis=0)
    regressors /= regressors.std(axis=0)

    return (U0, sv, Vsv, ypos, xpos, regressors, 
            behav_idx, stim_times, reward_times, stim_labels)


def load_visual_data(filename, stim_filename):
    # load neural activity
    dat = np.load(filename, allow_pickle=True)
    spks = zscore(dat["spks"], axis=1)
    istim = dat["istim"].astype("int")
    stim_times = dat["frame_start"]
    xpos, ypos = dat["xpos"], dat["ypos"]
    run = (dat["run"]**2).sum(axis=-1)**0.5

    # preprocess image data 
    dstim = loadmat(stim_filename, squeeze_me=True)
    np.random.seed(20)
    inds = np.random.randint(5000, size=(5,))
    print(inds)
    ex_stim = cv2.resize(dstim["img"][:,:,inds], (256,64)).transpose(2,0,1)

    nb = 100
    Ly, Lx = 24, 96
    img = np.zeros((5000, Ly, Lx))
    for j in range(5000//nb):
        img[j*nb:(j+1)*nb] = cv2.resize(dstim["img"][:,:,j*nb:(j+1)*nb], (Lx, Ly)).transpose((2,0,1))

    # all the receptive fields are on the front left monitor
    img = img[:, :, 36:-16]
    img -= img.mean(axis=0) # center the data

    nimg, Ly, Lx = img.shape
    print(img.shape)

    # SVD to keep top 200 components
    model_img = TruncatedSVD(n_components = 200).fit(np.reshape(img, (5000,-1)))
    img_U = model_img.components_
    img_pca = img.reshape(5000,-1) @ img_U.T

    return spks, istim, stim_times, xpos, ypos, run, ex_stim, img_pca, img_U, Ly, Lx

def load_alexnet_data(filename):
    dan = np.load(filename)
    sresp = []
    ilayer = []
    ipos = []
    iconv = []
    nmax = 1280*2
    np.random.seed(0)
    for j in range(1,6):
        conv = dan[f"conv{j}"][:5000]
        # crop edges
        nc, Ly, Lx = conv.shape[1:]
        crop = int(np.ceil(Lx/7))
        conv = conv[:,:,:,crop:-crop]
        nc, Ly, Lx = conv.shape[1:]
        print(conv.shape)
        xx, yy = np.meshgrid(np.arange(0, Ly), 
                            np.arange(0, Lx), indexing="ij")
        xx, yy = xx.flatten(), yy.flatten()
        S = conv.reshape(5000,-1).T.astype("float32")
        igood = np.nonzero(S.std(axis=1)>0)[0]
        S = S[igood]
        irand = np.random.permutation(S.shape[0])[:min(S.shape[0], nmax)]
        S = S[irand]
        inds = igood[irand] % (Ly * Lx)
        ipos.append(np.stack((xx[inds], yy[inds]), axis=1))
        iconv.append(igood[irand] // (Ly * Lx))
        sresp.append(S)
        ilayer.append((j-1)*np.ones(len(S)))
    sresp = np.vstack(tuple(sresp))
    ilayer = np.hstack(tuple(ilayer))
    ipos = np.vstack(tuple(ipos))
    iconv = np.hstack(tuple(iconv))

    return sresp, ilayer, ipos, iconv, nmax
