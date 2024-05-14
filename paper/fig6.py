"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import zscore
from sklearn.decomposition import PCA
import cv2

from fig_utils import *

def fig6(root, alldat, save_figure=True):
    d = np.load(os.path.join(root, "results/steinmetz_proc.npz"), allow_pickle=True)
    ccf_all = d["ccf_all"]
    itrials = d["itrials"]
    isorts = d["isorts"]
    reaction_times = d["reaction_times"]
    face_motions = d["face_motions"]
    wheel_moves = d["wheel_moves"]
    pupil_speeds = d["pupil_speeds"]
    perc = d["perc"]
    area = d["area"]
    licks = d["licks"]
    rewards = d["rewards"]

    try:
        im_b = cv2.imread(os.path.join(root, "figures/ccf_brain.jpg"))
        im_b = cv2.resize(im_b, (im_b.shape[1]//5, im_b.shape[0]//5))
        im_b = im_b.mean(axis=-1)
        im_s = cv2.imread(os.path.join(root, "figures/ibl_schematic.png"))
        im_s = cv2.cvtColor(im_s, cv2.COLOR_BGR2RGB)
    except:
        print("missing schematic figs")

    d = 26
    dat = alldat[d]
    itrial_ex = itrials[2*d]
    isort_ex = isorts[2*d]

    fig = plt.figure(figsize=(14,10), dpi=100)
    grid = plt.GridSpec(4,6, figure=fig, left=0.05, right=0.98, top=0.96, bottom=0.09, 
                            wspace = 0.4, hspace = 0.4)
    transl = mtransforms.ScaledTranslation(-15 / 72, 33 / 72, fig.dpi_scale_trans)
    il = 0

    ax = plt.subplot(grid[0,0])
    poss = ax.get_position().bounds
    ax.set_position([poss[0]-0.025, *poss[1:]])
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.imshow(im_s)
    ax.axis("off")
    ax.set_title("visual discrimination\n(Steinmetz et al, 2019)", fontsize="medium")

    ax = plt.subplot(grid[1,0])
    poss = ax.get_position().bounds
    ax.set_position([poss[0]-0.025, poss[1]+0.015, *poss[2:]])
    ax.imshow(im_b<255, cmap="gray_r", vmax=5)
    xpos = (ccf_all[:,0]*0.95+820)/5
    ypos = (ccf_all[:,1]*0.85+400)/5
    ax.scatter(xpos, ypos, color="y", s=1, alpha=0.1, rasterized=True)#c=model.embedding[::10,0], s=3, alpha=0.1, 
    ax.set_xlim([0, 11000//5])
    ax.axis("off")
    ax.plot([-2,2])
    ax.set_title("neuropixels recordings\n(10 mice, 39 sessions)", fontsize="medium")
    ax.plot(200 + 250*np.arange(0, 2), np.ones(2)*1500, color="k")
    ax.text(150, 1500, "A", fontsize="small", ha="right", va="center")
    ax.text(500, 1500, "P", fontsize="small", ha="left", va="center")

    rts_ex = (dat["reaction_time"][itrial_ex,0]/10).astype("int") + 50
    fb_ex  = (dat["feedback_time"][itrial_ex]*100) + 50
    fb_ex[dat["feedback_type"][itrial_ex]==-1] = np.nan
    fb_ex[fb_ex>250] = np.nan

    spks = dat["spks"].copy().astype("float32")
    nn,_,nt = spks.shape
    brain_area = dat["brain_area"]
    spks = spks.reshape(nn, -1)
    igood = ((spks.mean(axis=-1)) / .01) > 0.1
    igood *= (brain_area != "root")
    iexs = [204, 210]
    spk_ex = [dat["spks"][igood][iex].copy() for iex in iexs]
    brain_area_ex = [brain_area[igood][iex] for iex in iexs]
    spks = zscore(spks[igood], axis=-1)
    n_PCs = 10
    spcs = PCA(n_components=n_PCs).fit_transform(spks.T).T
    U = spks @ (spcs.T / (spcs**2).sum(axis=1)**0.5)
    spcs_trials = spcs.reshape(n_PCs, -1, nt)
    sresp = spcs_trials[:,itrial_ex].copy()
    sresp -= sresp.mean(axis=(-2,-1), keepdims=True)
    sresp_std = sresp.std(axis=(-2,-1), keepdims=True)
    sresp /= sresp_std

    transl = mtransforms.ScaledTranslation(-25 / 72, 6 / 72, fig.dpi_scale_trans)
    #ax = plt.subplot(grid[:2, 1:3])
    #grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=ax, 
    #                                                        wspace=0.3, hspace=0.45)
    #ax.remove()
    for k in range(len(iexs)):
        for j in range(2):
            ax = plt.subplot(grid[j, k+1])
            poss = ax.get_position().bounds
            ax.set_position([poss[0]-0.01, *poss[1:]])
            if j==0:
                ax.imshow(spk_ex[k][itrial_ex], aspect="auto", cmap="gray_r", vmax=1, vmin=0)
                ax.set_title(f"example neuron ({brain_area_ex[k]})", fontsize="medium")
                if k==0:
                    il = plot_label(ltr, il, ax, transl, fs_title)
                    ax.set_ylabel("trials")
                    ax.set_xlabel("time from stimulus (sec.)\n ")
                #ax.text(0.5, -.35, "time from stimulus (sec.)", transform=ax.transAxes, ha="center")  
            else:
                ax.imshow(spk_ex[k][itrial_ex][isort_ex], aspect="auto", cmap="gray_r", vmax=1, vmin=0)
                if k==0:
                    ax.set_ylabel("trials (sorted by\nrastermap)")
            ax.set_yticks([])
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.set_xticks([50, 150, 250])
            ax.set_xlim([0, 250])
            ax.set_xticklabels(["0", "1", "2"])

    ss = sresp[:, isort_ex].copy()
    ss /= ss.std(axis=(-2,-1), keepdims=True)
    ax = plt.subplot(grid[:2, 3:5])
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=ax, 
                                                            wspace=0.3, hspace=0.45)
    ax.remove()
    for i in range(4):
        ax = plt.subplot(grid[i//2, i%2+3])
        poss = ax.get_position().bounds
        ax.set_position([poss[0]-0.01-0.005*(i%2), *poss[1:]])
        im = ax.imshow(ss[i], aspect="auto", vmin=-1, vmax=1, cmap="magma")
        if i==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            ax.set_ylabel("trials (sorted by\nrastermap)")
            poss = ax.get_position().bounds
            cax = fig.add_axes([poss[0]+1.03*poss[2], poss[1] + poss[3]*0.3, 
                                poss[3]*0.03, 0.3*poss[3]])
            plt.colorbar(im, cax)
            ax.set_xlabel("time from stimulus (sec.)\n ")
        ax.set_title(f"PC {i+1}", fontsize="medium")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xticks([50, 150, 250])
        ax.set_xticklabels(["0", "1", "2"])

    transl = mtransforms.ScaledTranslation(-40 / 72, 6 / 72, fig.dpi_scale_trans)
    
    labels = ["reaction time (sec.)", "reward",  "licking", "wheel movement", 
              "face motion", "pupil speed", "trial # (norm.)", ""]
    jy = [0, 1, 3, 3, 3, 3, 2, 3]
    jx = [5, 5, 0, 1, 2, 3, 4, 4]
    #jy = [1, 1, 1, 1, 0, 1, 2, 3]
    #jx = [1, 2, 3, 4, 5, 5, 0, 0]
    for j,y in enumerate([reaction_times, rewards, licks, wheel_moves, face_motions, 
                          pupil_speeds, isorts, isorts]):
        ax = plt.subplot(grid[jy[j], jx[j]])
        poss = ax.get_position().bounds
        if j>1 and j<6:
            ax.set_position([poss[0]-(j-2)*0.008-0.01, poss[1]+0.01, *poss[2:]])
        elif j==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            il+=1
        elif j==6:
            il = plot_label(ltr, il, ax, transl, fs_title)
            il-=2
        #elif j==5:
        #    ax.set_position([poss[0], poss[1]+0.01, *poss[2:]])
        
        #if j>0:
        #    ax.set_position([poss[0]-0.01*(j%4) + 0.01*(j!=1), poss[1] - (j//4)*0.035, *poss[2:]])
        #if j==0 or j==2 or j==3 or j==4:
        #    if j>0:
        #        transl = mtransforms.ScaledTranslation(-30 / 72, 6 / 72, fig.dpi_scale_trans)
        #    il = plot_label(ltr, il, ax, transl, fs_title)
        nbins = 10
        rt0 = np.zeros((len(y), nbins+1))
        for k in np.arange(0, len(y), 1):
            rt = y[k].copy()
            if j>5:
                itrial = np.nonzero(itrials[k])[0]
                ntot = len(itrials[k])
                if j==6:
                    rt = rt[::-1] if isorts[k][:10].mean() > isorts[k][-10:].mean() else rt
                    rt = itrial[rt] / ntot # normalize sorting
                else:
                    rt = itrial[rt] / ntot # normalize sorting
                    iroll = np.random.randint(len(rt))
                    rt = np.roll(rt, iroll) # random roll
                    isort0 = np.roll(isorts[k].copy(), iroll)
                    rt = rt[::-1] if isort0[:10].mean() > isort0[-10:].mean() else rt
            elif j==1:
                rt = (rt > 0).astype("int") # set reward to 1 and no-reward to 0
            elif j==3:
                rt = np.abs(rt) # abs value of wheel movements (left or right)
            elif j==5:
                rt[rt > 0.4] = np.nan # remove pupil outliers
            xt = np.linspace(0, 1, len(rt))
            if j<6 and j!=1:
                ax.scatter(xt, rt, color=0.5*np.ones(3), alpha=0.25, 
                           s=0.5, rasterized=True)
            elif j>5:
                ax.plot(xt, rt, color=0.5*np.ones(3), alpha=0.25, lw=0.5)
            ib = np.round(xt / (1/nbins)).astype(int)
            rtb = np.array([rt[ib==i].mean() for i in range(nbins+1)])
            xb = np.arange(0, 1+1/nbins, 1/nbins)
            rt0[k] = rtb

        ax.errorbar(xb, np.nanmean(rt0, axis=0), np.nanstd(rt0, axis=0)/(38**0.5), color="k")
        if j==6:
            ax.set_title("Rastermap sorting", fontsize="medium", ha="center", y=0.95, loc="center")
        elif j==7:
            ax.set_title("random shuffle", fontsize="medium", ha="center", y=0.95, loc="center")
            #ax.set_yticklabels([])
        if j==0 or j==2 or j==6:
            ax.set_xlabel("trial embedding (norm.)")
        #if j>1:
        #    ax.set_title(labels[j], fontsize="medium", ha="center", y=0.95, loc="center")
        #elif j==0:
        ax.set_ylabel(labels[j])
        if j==1:
            ax.set_ylim([0, 1])
            ax.set_yticks(np.arange(0, 1.1, 0.2))
        elif j==3:
            ax.set_yticks([0, 5, 10])
        elif j>5:
            ax.set_yticks([0, 0.5, 1])

    transl = mtransforms.ScaledTranslation(-25 / 72, 6 / 72, fig.dpi_scale_trans)
    
    pspeed = (np.diff(dat["pupil"][1:], axis=-1)**2).sum(axis=0)**0.5
    cmaps = ["gray_r", "PiYG", "viridis", "viridis"]
    vmaxs = [1, 5, 4, 0.2]
    titles = ["licks", "wheel movement (a.u.)", "face motion (a.u.)", "pupil speed (a.u.)"]
    for j, y in enumerate([dat["licks"].squeeze(), dat["wheel"].squeeze(), dat["face"].squeeze(), pspeed]):
        ax = plt.subplot(grid[2, j])
        ax.set_title(titles[j], fontsize="medium")
        poss = ax.get_position().bounds
        ax.set_position([poss[0]-j*0.008-0.01, *poss[1:]])
        ax.set_yticks([])
        ax.set_xticks([50, 150, 250])
        ax.set_xlim([0, 250])
        ax.set_xticklabels(["0", "1", "2"])
        if j>0:
            if j!=1:
                ax.spines["left"].set_visible(False)
            else:
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
            im = ax.imshow(y[itrial_ex][isort_ex], aspect="auto", vmin=0 if j!=1 else -vmaxs[j], vmax=vmaxs[j], cmap=cmaps[j])
            poss = ax.get_position().bounds
            cax = fig.add_axes([poss[0]+1.03*poss[2], poss[1] + poss[3]*0.3, 
                                poss[3]*0.03, 0.3*poss[3]])
            plt.colorbar(im, cax)
            if j==1:
                cax.text(1.03, 0.68, "left", transform=ax.transAxes, fontstyle="italic")
                cax.text(1.03, 0.13, "right", transform=ax.transAxes, fontstyle="italic")
        else:
            ax.set_xlabel("time from stimulus (sec.)")
            il = plot_label(ltr, il, ax, transl, fs_title)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            lt = np.nonzero(y[itrial_ex][isort_ex])
            ax.scatter(lt[1], lt[0], s=1, color="k", alpha=1)
            ax.scatter(rts_ex[isort_ex], np.arange(0, itrial_ex.sum()), s=7, 
                       color="m", alpha=1, marker="x", lw=0.5)
            ax.scatter(fb_ex[isort_ex], np.arange(0, itrial_ex.sum()), s=7, 
                       color="b", alpha=1, marker="x", lw=0.5)
            ax.text(0.2, 1.2, "wheel move start", color="m", transform=ax.transAxes)
            ax.text(0.75, 1.08, "reward", color="b", transform=ax.transAxes)
            ax.set_ylabel("trials (sorted by\nrastermap)")


    #ax = plt.subplot(grid[2:, -2:])
    #grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=ax, 
    #                                                        wspace=0.5, hspace=0.8)
    #ax.remove()

    il +=1
    ax = plt.subplot(grid[2:,-1])
    il = plot_label(ltr, il, ax, transl, fs_title)
    cols = ["r", "b"]
    for k in range(len(perc)):
        ax.scatter(np.arange(0,7)+0.1*np.random.randn(7), perc[k,0], color="r", s=10, alpha=0.25)#, marker=m[k%2])
        ax.scatter(np.arange(0,7)+0.1*np.random.randn(7), perc[k,1], color="b", s=10, alpha=0.25)#, marker=m[k%2])
    for j in range(2):
        ax.errorbar(np.arange(0, 7), np.nanmean(perc[:,j], axis=0), 
                np.nanstd(perc[:,j], axis=0) / ((~np.isnan(perc[:,j])).sum(axis=0)-1)**0.5, 
                color=cols[j], lw=3)
    ax.set_ylabel("% of neurons")
    plt.xticks(np.arange(0, 7))
    ax.set_xticklabels(area, rotation=45, ha="right")
    ax.set_ylim([0, 86])
    ax.text(1, 0.95, "late-active", color="b", ha="right", transform=ax.transAxes)
    ax.text(1, 0.89, "early-active", color="r", ha="right", transform=ax.transAxes)

    if save_figure:
        fig.savefig(os.path.join(root, "figures", "fig6.pdf"), dpi=150)