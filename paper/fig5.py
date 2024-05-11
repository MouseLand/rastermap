"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import patches

from fig_utils import *


def fig5(root, psths, rts, gts, save_figure=True):    
    il = 0

    nk, nn, nt2 = psths.shape
    nt = nt2//2

    ci = np.hstack((np.linspace(0, 0.3, 5), np.linspace(0.7, 1., 5)))
    colors = plt.get_cmap("PiYG")(ci)
    c0 = colors[np.newaxis,:].copy()
    c0[:, 5:] = 1.0
    c1 = colors[np.newaxis,1:].copy()
    c1[:, :4] = 1.0
    cis = [c0, c1]

    c_t = plt.get_cmap("YlOrBr")([0.4, 0.6, 0.9])[::-1]

    fig = plt.figure(figsize=(14,6))

    grid = plt.GridSpec(2,7, figure=fig, left=0.03, right=0.98, top=0.85, bottom=0.05, 
                            wspace = 0.2, hspace = 0.3)

    t_sample = np.hstack((np.linspace(480, 800, 5), 
                        np.linspace(800, 1200, 5))).astype("int")

    tstr = [r"short prior block, $t_s$ (ms)", r"long prior block, $t_s$ (ms)"]

    ax0 = plt.subplot(grid[0,:2])
    pos = ax0.get_position().bounds
    ax0.set_position([pos[0], pos[1], pos[2]*0.85, pos[3]])
    transl = mtransforms.ScaledTranslation(-18 / 72, 40 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax0, transl, fs_title)
    ax0.axis("off")
            
    ax = ax0.inset_axes([0., 0.8, 1., 0.15])
    ax.plot([0, 1.2], [0, 0], ls="-", color="k")
    ax.plot([0, 0], [-1, 1], color="k")
    ax.plot([1, 1], [-1, 1], color="k")
    ax.plot([1.2, 1.8], [0, 0], ls="dotted", color="k")
    ax.plot([1.8, 2], [0, 0], ls="-", color="k")
    ax.plot([2, 2], [-1, 1], color="k")
    ax.text(0, 1.1, "ready cue", va="bottom", ha="center", color=c_t[0])
    ax.text(0.5, -1.5, r"$t_s$", va="bottom", ha="center")
    ax.text(1, 1.1, "set cue", va="bottom", ha="center", color=c_t[1])
    ax.text(1.5, -1.5, r"$t_p$", va="bottom", ha="center")
    ax.text(2, 1.1, "go (action)", va="bottom", ha="center", color=c_t[2])
    ax.text(1.65, -3.5, r"reward $\propto$ |$t_p$ - $t_s$| / $t_s$", va="bottom", ha="center")
    ax.text(1, 3.5, "time-interval reproduction task\n(Sohn, Narain et al, 2019)", fontsize="large",
            va="bottom", ha="center")
    ax.set_xlim([-0.05,2.05])
    ax.set_ylim([-1,1])
    ax.axis("off")

    for j in range(2):
        ax = ax0.inset_axes([0., 0.35-0.3*j, 1., 0.125])
        ax.imshow(cis[j], aspect="auto")
        ax.axis("off")
        for i in range(5):
            ax.text(i+4*j, 0, t_sample[i+5*j], ha="center", va="center", fontsize="small",
                    color="w", fontweight="bold")
            if (i==4 and j==0) or (i==0 and j==1):
                width = 1
                ax.add_patch(patches.Rectangle(xy=(i+4*j-0.55, -0.5), width=width, 
                            fill=False, height=1, facecolor=None, edgecolor="k", lw=3))
        ax.text(j*4+2, -0.8, tstr[j], ha="center", fontsize="large")
        ax.set_xlim([-0.5, 8.5])

    kis = np.hstack((np.arange(4,-1, -1), np.arange(5, 10)))
    for k in range(10):
        ki = kis[k]
        ax = plt.subplot(grid[ki//5, 2+ki%5])
        pos = ax.get_position().bounds
        im = ax.imshow(psths[k], aspect="auto", vmin=0, vmax=5, cmap="gray_r")
        for l,tt in enumerate([rts[k], nt, gts[k]]):
            ax.plot([tt, tt], [0, nn], ls="--", lw=2, color=c_t[l])
        ax.set_ylim([0, nn])
        ax.set_xlim([nt-80, nt+80])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title(f"{t_sample[k]} ms", color=colors[k], fontweight="bold", 
                    loc="center", fontsize="medium")
        #ax.imshow(psth_s - psth_l, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
        if ki%5 == 2:
            ax.text(0.5, 1.15, ["short prior block", "long prior block"][k//5], 
                    transform=ax.transAxes, fontsize="large", ha="center")
        if ki==0:
            ax.text(-0.01, 1.22, "trial-averaged responses", 
                    transform=ax.transAxes, fontsize="large")
            transl = mtransforms.ScaledTranslation(-18 / 72, 40 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl, fs_title)
        if ki==1:
            cax = fig.add_axes([pos[0]+0.*pos[2], pos[1] - pos[3]*0.08, 
                                    pos[2]*0.5, 0.03*pos[3]])
            plt.colorbar(im, cax, orientation="horizontal")
            

    ax = plt.subplot(grid[1,0])
    transl = mtransforms.ScaledTranslation(-50 / 72, 10 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.06, pos[1], pos[2], pos[3]])
    pos = ax.get_position().bounds
    im = ax.imshow(psths[4] - psths[5], aspect="auto", vmin=-5, vmax=5, cmap="RdBu_r")
    k = 4
    for l,tt in enumerate([rts[k], nt, gts[k]]):
        ax.plot([tt, tt], [0, nn], ls="--", lw=2, color=c_t[l])
    ax.set_ylim([0, nn])
    ax.set_xlim([nt-80, nt+80])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.set_title(r"short - long prior, $t_p$ = 800 ms", loc="center")
    cax = fig.add_axes([pos[0] + 1.1*pos[2], pos[1] + pos[3]*0.65, 
                        pos[2]*0.04, 0.3*pos[3]])
    plt.colorbar(im, cax)
            
    if save_figure:
        fig.savefig(os.path.join(root, "figures", "fig5.pdf"), dpi=200)