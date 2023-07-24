"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import matplotlib.pyplot as plt 
from matplotlib import patches
import os
import numpy as np
from fig_utils import *

ccolor = [[0,1,0], [0,0,0.8]]

def panel_neuron_pos(fig, grid, il, yratio, xpos0, ypos0, isort, brain_img):
    xpos, ypos = xpos0.copy(), -1*ypos0.copy()
    ylim = np.array([ypos.min(), ypos.max()])
    xlim = np.array([xpos.min(), xpos.max()])
    ylr = np.diff(ylim)[0] / np.diff(xlim)[0]
    
    ax = fig.add_subplot(grid[0,0])
    poss = ax.get_position().bounds
    ax.set_position([poss[0]+0.01, poss[1]-.04, 1*poss[2], 
                      1*poss[2] / ylr * yratio])
    poss = ax.get_position().bounds
    
    memb = np.zeros_like(isort)
    memb[isort] = np.arange(0, len(isort))
    subsample = 5
    ax.scatter(ypos[::subsample], xpos[::subsample], cmap=cmap_emb, 
                s=0.5, alpha=0.5, c=memb[::subsample], rasterized=True)
    ax.axis("off")
    
    add_apml(ax, xpos, ypos)

    axin = fig.add_axes([poss[0]-0.02, poss[1] +poss[3]*.8, 
                            poss[2]*0.3, poss[3]*0.3])
    axin.imshow(brain_img)
    axin.axis("off")
    transl = mtransforms.ScaledTranslation(-8 / 72, -0/ 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, axin, transl, fs_title)
    
    return il

def panels_tuning(axs, il, padding, corridor_tuning, label_white=True):
    nov = 30
    n_corr, nn, npts = corridor_tuning.shape
    for icorr in range(n_corr):
        ctmax = corridor_tuning[icorr].max()
        ctmin = corridor_tuning[icorr].min()
        npl = 100
        ipl = np.linspace(1, nn-npl//4, npl).astype("int")
        for i in ipl:
            ct = corridor_tuning[icorr, i].copy()
            ct -= ctmin
            ct /= ctmax
            axs[icorr].plot(np.arange(0, npts), i - ct*nov + nov/2, #(n_sn-i-24)+ct*nov, 
                        color=ccolor[icorr], lw=0.5)
        axs[icorr].plot((npts*2/3) * np.ones(2), [0, nn*(1+padding)], 
                        color='k', lw=1, zorder=5)
        if label_white:
            axs[icorr].text(2/3 + 0.02, 0.02, 'white space start', 
                        transform=axs[icorr].transAxes, va='bottom', rotation=90)
        if icorr==0:
            axs[icorr].set_title("tuning curves")
            #text(0, 1, 'tuning curves', ha='left', 
                       # transform=axs[icorr].transAxes, fontsize="large")
            axs[icorr].text(1.1, -0.05, "position (cm)", ha="center", va="top",
                    transform=axs[icorr].transAxes)
            transl = mtransforms.ScaledTranslation(-15 / 72, 5/ 72, axs[icorr].figure.dpi_scale_trans)
            il = plot_label(ltr, il, axs[icorr], transl, fs_title)

        axs[icorr].set_xlim([0, npts])
        axs[icorr].set_ylim([0, nn*(1+padding)])
        axs[icorr].invert_yaxis()
        axs[icorr].spines["left"].set_visible(False)
        axs[icorr].set_yticks([])
        axs[icorr].set_xticks([0, 2/3*100])
        axs[icorr].set_xticklabels(["0", "40"])
    return il

def panel_raster(fig, ax, il, padding, sn, xmin, xmax, 
            corridor_starts, corridor_widths, reward_inds, 
            cmap_neurons=True, 
            title_str="neural activity in virtual reality"):
    poss = ax.get_position().bounds
    cax = fig.add_axes([poss[0]-0.035, poss[1]+poss[3]-0.12*poss[3], 
                        poss[3]*0.005, 0.1*poss[3]])
    plot_raster(ax, sn, xmin=xmin, xmax=xmax, 
                vmax=2, fs=3.38, n_neurons=5000, nper=100, label=True, 
                padding=padding, cax=cax, cax_label="left",
                cax_orientation="vertical", label_pos="right") 
    #plt.colorbar(im, cax, orientation="horizontal")
    #cax.set_xlabel("z-scored\n ")   
    ax.set_title(title_str)
    transl = mtransforms.ScaledTranslation(-15 / 72, 5/ 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)

    nn = sn.shape[0]
    if cmap_neurons:
        cax = fig.add_axes([poss[0]-poss[2]*0.02, poss[1], poss[2]*0.01, poss[3]])
        cols = cmap_emb(np.linspace(0, 1, nn))
        cax.imshow(cols[:,np.newaxis], aspect="auto")
        cax.set_ylim([0, (1+padding)*nn])
        cax.invert_yaxis()
        cax.axis("off")

    # add corridor colors
    for n in range(len(corridor_starts)):
        if (corridor_starts[n,0]+corridor_widths[n] > xmin and 
                corridor_starts[n,0] < xmax):
            icorr = int(corridor_starts[n,1])
            start = corridor_starts[n,0]
            width = corridor_widths[n]
            width += min(0, start-xmin)
            start = max(0, start - xmin)
            width = min(width, xmax - xmin - start)
            ax.add_patch(
                patches.Rectangle(xy=(start, 0), width=width,
                            height=nn, facecolor=ccolor[icorr], 
                            edgecolor=None, alpha=0.1))
    # add reward events
    for n in range(len(reward_inds)):
        if reward_inds[n] > xmin and reward_inds[n] < xmax:
            start = int(reward_inds[n] - xmin)
            width = 0
            ax.add_patch(patches.Rectangle(xy=(start, 0), width=width,
                        height=nn, facecolor=None, edgecolor='g', alpha=1))

    return il

def panel_events(ax, xmin, xmax, sound_inds, lick_inds, reward_inds):
    h1=ax.scatter(sound_inds-0.5,0*np.ones([len(sound_inds),]), 
                        color=[1.,0.6,0], marker='s', s=30)
    h2=ax.scatter(lick_inds-0.5,-1*np.ones([len(lick_inds),]), 
                        color=[1.0,0.3,0.3], marker='.', s=30)
    h0=ax.scatter(reward_inds-0.5,1*np.ones([len(reward_inds),]), 
                        color='g', marker='^', s=30)
    ax.axis('off')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-1.35, 1.35])
    ax.legend([h0,h1,h2], ["reward", "tone", "licks"], 
                handletextpad=0.01, labelspacing=0.15, loc=(-0.08,-0.31), 
                labelcolor="linecolor", frameon=False)

def panel_imgs(grid, il, corridor_imgs):
    Ly, Lx = corridor_imgs.shape[1:]
    Lyc = Lx*4
    xp = int(Lx*0.4)
    imgs = 255*np.ones((Lx*2+xp*2, Lyc), "uint8")
    for k in range(2):
        imgs[(Lx+xp)*k+xp : (Lx+xp)*k+xp + Lx] = corridor_imgs[k, :Lyc].T
    imgs = np.tile(imgs[:,:,np.newaxis], (1,1,3))

    ax = plt.subplot(grid[1,0])
    ax.imshow(imgs)
    for k in range(2):
        ax.text(0, (Lx+xp)*k + xp-10, "leaves" if k==0 else "circles", 
                color=ccolor[k])
    ax.axis("off")
    ax.set_title("VR corridors")
    transl = mtransforms.ScaledTranslation(-15 / 72, 5/ 72, grid.figure.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    
    return il

def panel_cc(grid, il, yratio, cc_nodes):
    ax = plt.subplot(grid[-1, 0])
    poss = ax.get_position().bounds
    ax.set_position([poss[0], poss[1]-.0, 0.95*poss[2], 
                      0.95*poss[2] * yratio])
    poss = ax.get_position().bounds
    vmax = 1
    im = ax.imshow(cc_nodes, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.axis("off")
    ax.set_title("asymmetric similarity")
    transl = mtransforms.ScaledTranslation(-15 / 72, 5/ 72, grid.figure.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    
    cax = grid.figure.add_axes([poss[0]+poss[2]*1.02, poss[1]+poss[3]*0.75, 
                        poss[2]*0.03, poss[3]*0.25])
    plt.colorbar(im, cax)
    return il

def _fig3(brain_img, sn, xpos, ypos, isort, isort2, cc_nodes,
         corridor_starts, corridor_widths, 
         corridor_tuning, corridor_imgs, VRpos,
         reward_inds, sound_inds, lick_inds, run):
    fig = plt.figure(figsize=(14,7))
    yratio = 14 / 7
    grid = plt.GridSpec(3,5, figure=fig, left=0.02, right=0.98, top=0.98, bottom=0.02, 
                        wspace = 0.3, hspace = 0.15)

    il = 0
    il = panel_neuron_pos(fig, grid, il, yratio, xpos, ypos, isort, brain_img)

    il = panel_imgs(grid, il, corridor_imgs)

    il = panel_cc(grid, il, yratio, cc_nodes)
    
    ax = plt.subplot(grid[:,1:])
    pos = ax.get_position().bounds
    ax.remove()

    xmin = 1410
    xmax=xmin+520

    nn = sn.shape[0]
    xr = xmax - xmin
    y0 = pos[1]
    x0 = pos[0]
    padding=0.025
    dye = 0.06
    dyr = 0.09
    dx = 0.8
    xpad = 0.03*pos[2]
    xpadt = 0.01*pos[2]
    dxt = ((1-dx)*pos[2]-xpad-xpadt)/2
    ypad = 0.02*pos[3]
    ys = y0+(dye+dyr)*pos[3]+ypad+0.01*pos[3]
    poss = [x0, ys, pos[2]*dx, pos[3]-ys]

    ax = fig.add_axes(poss)
    il = panel_raster(fig, ax, il, padding, sn, xmin, xmax, 
                    corridor_starts, corridor_widths, reward_inds)

    ax = fig.add_axes([poss[0], y0+dyr*pos[3]+ypad, poss[2], dye*pos[3]])
    panel_events(ax, xmin, xmax, sound_inds, lick_inds, reward_inds)

    ax = fig.add_axes([poss[0], y0, poss[2], dyr*pos[3]])
    ax.fill_between(np.arange(0, xr), run[xmin:xmax], color=kp_colors[0])
    ax.set_xlim([0, xr])
    ax.set_ylim([0, np.percentile(run[xmin:xmax], 99)])
    ax.axis("off")
    ax.text(0.11,0.9,"running speed", transform=ax.transAxes, color=kp_colors[0])
            
    axs = [fig.add_axes([poss[0]+poss[2]+xpad, poss[1], dxt, poss[3]]),
        fig.add_axes([poss[0]+poss[2]+xpad+xpadt+dxt, poss[1], dxt, poss[3]])]
    
    il = panels_tuning(axs, il, padding, corridor_tuning)

    return fig


def fig3(root, save_figure=True):
    d = np.load(os.path.join(root, "results", "corridor_proc.npz"), allow_pickle=True) 
    d2 = np.load(os.path.join(root, "data", "corridor_behavior.npz"), allow_pickle=True) 
    try:
        brain_img = plt.imread(os.path.join(root, "figures", "brain_window_visual.png"))
    except:
        brain_img = np.zeros((50,50))

    fig = _fig3(brain_img, **d, **d2)
    if save_figure:
        fig.savefig(os.path.join(root, "figures", "fig3.pdf"), dpi=200)

          
def _suppfig_vr_algs(snys, ctunings,
        corridor_starts, corridor_widths, reward_inds):
        
    fig = plt.figure(figsize=(12,12))
    grid = plt.GridSpec(2,1, figure=fig, left=0.06, right=0.96, top=0.96, bottom=0.04, 
                            wspace = 0.3, hspace = 0.15)

    xmin = 1000
    xmax=xmin+500
    il = 0
    padding = 0.025
    alg = ["t-SNE", "UMAP"]
    for k in range(2):
        sny = snys[k]
        ctuning = ctunings[k]

        ax = plt.subplot(grid[k])
        pos = ax.get_position().bounds
        ax.remove()

        xmin = 1000
        xmax=xmin+500

        nn = sny.shape[0]
        xr = xmax - xmin
        y0 = pos[1]
        x0 = pos[0]
        padding=0.025
        dx = 0.8
        xpad = 0.03*pos[2]
        xpadt = 0.01*pos[2]
        dxt = ((1-dx)*pos[2]-xpad-xpadt)/2
        poss = [x0, y0, pos[2]*dx, pos[3]]

        ax = fig.add_axes(poss)
        il = panel_raster(fig, ax, il, padding, sny, xmin, xmax, 
                corridor_starts, corridor_widths, reward_inds, 
                cmap_neurons=False, title_str=f"{alg[k]} sorting")
        axs = [fig.add_axes([poss[0]+poss[2]+xpad, poss[1], dxt, poss[3]]),
            fig.add_axes([poss[0]+poss[2]+xpad+xpadt+dxt, poss[1], dxt, poss[3]])]
        il = panels_tuning(axs, il, padding, ctuning, label_white=False)
    return fig

def suppfig_vr_algs(root, save_figure=True):
    d = np.load(os.path.join(root, "results", "corridor_supp.npz"), allow_pickle=True) 
    fig = _suppfig_vr_algs(**d);
    if save_figure:
        fig.savefig(os.path.join(root, "figures", "suppfig_vr_algs.pdf"))
