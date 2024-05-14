"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import matplotlib.pyplot as plt 
from matplotlib import patches
import os
import numpy as np
from fig_utils import *

def panel_neuron_pos(fig, grid1, il, yratio, xpos0, ypos0, isort, brain_img):
    xpos, ypos = xpos0.copy(), -1*ypos0.copy()
    ylim = np.array([ypos.min(), ypos.max()])
    xlim = np.array([xpos.min(), xpos.max()])
    ylr = np.diff(ylim)[0] / np.diff(xlim)[0]
    
    ax = fig.add_subplot(grid1[:1])
    poss = ax.get_position().bounds
    ax.set_position([poss[0]-0.01, poss[1]-.11, 1.4*poss[2], 1.4*poss[2]/ylr * yratio])
    poss = ax.get_position().bounds
    transl = mtransforms.ScaledTranslation(-10 / 72, -12 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    memb = np.zeros_like(isort)
    memb[isort] = np.arange(0, len(isort))
    subsample = 5
    ax.scatter(ypos[::subsample], xpos[::subsample], cmap=cmap_emb, 
                s=0.5, alpha=0.5, c=memb[::subsample], rasterized=True)
    ax.axis("off")
    add_apml(ax, xpos, ypos)

    axin = fig.add_axes([poss[0]+poss[2]*0.5, poss[1] +poss[3]*.6, poss[2]*0.5, poss[3]*0.5])
    axin.imshow(brain_img)
    axin.axis("off")
    return il

def panels_beh_traces(grid1, il, face_img, beh, beh_names, tcam, tneural, itest, xmin, xmax):
    ax = plt.subplot(grid1[1])
    poss = ax.get_position().bounds
    ax.set_position([poss[0]-0.01, poss[1]-0.07, 1.4*poss[2], 1.4*poss[3]])
    ax.imshow(face_img, vmin=100, vmax=150)
    ax.set_title("   behaviors")
    ax.axis("off")
    transl = mtransforms.ScaledTranslation(-10 / 72, 7 / 72, grid1.figure.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)

    t0 = np.abs(tneural[itest.flatten()][xmin] - tcam).argmin()
    t1 = np.abs(tneural[itest.flatten()][xmax] - tcam).argmin()
    ax = plt.subplot(grid1[2])
    poss = ax.get_position().bounds
    ax.set_position([poss[0]-0.01, poss[1]-0.03, 1.4*poss[2], 1.6*poss[3]])
    for k in range(beh.shape[1]):
        by = beh[t0:t1, k].copy()
        if k==0:
            by = np.maximum(0, by)
        by -= by.min()
        by /= by.max()
        ax.plot(by - k*1.5, color=kp_colors[k], lw=0.5);
        ax.text(t1-t0, -k*1.5, beh_names[k], va="top", ha="right", color=kp_colors[k])
    ax.plot([0, 50*10], -1.5*k*np.ones(2), color="k")
    ax.text(0,0, "10 sec.", transform=ax.transAxes, va="top")
    ax.set_ylim([-1.5*k-0.1, 1])
    ax.set_xlim([0, t1-t0])
    ax.axis("off")
    return il

      
def panels_rfs(grid, il, yh, padding, ipl,rfs, beh_names):
    nn = rfs.shape[0]
    xw = 1
    ax = plt.subplot(grid[4])
    poss = ax.get_position().bounds
    ax.set_position([poss[0]+0.02, poss[1], poss[2], poss[3]*yh])
    transl = mtransforms.ScaledTranslation(-15 / 72, 38 / 72, grid.figure.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
        
    dy = 300
    dx = 1.
    l = np.array([0,1,2])
    npl = len(ipl)
    h = 4
    for i in range(npl):
        ir = ipl[i]
        rf = rfs[ir, 201-100:201+100].copy() / dx
        for k in range(rfs.shape[-1]):
            ax.plot(np.arange(0,rf.shape[-2]) + k*dy, rf[:,k]*-h + ir, color=kp_colors[k], lw=1)
    ax.set_ylim([0, nn*(1+padding)])
    ax.invert_yaxis()
    ax.axis("off")
    ax.text(0.01,1.09, "behavioral receptive fields", transform=ax.transAxes, fontsize="large")
    for k in range(len(kp_colors)):
        ax.text(k/len(kp_colors),1.005,beh_names[k], color=kp_colors[k], 
                rotation=45, transform=ax.transAxes, size="small")

    ax = plt.subplot(grid[5])
    poss = ax.get_position().bounds
    ax.set_position([poss[0]+0.0, poss[1], poss[2], poss[3]*yh])
    cmap_rb = plt.get_cmap("RdBu_r")
    cmap_rb.set_bad("white")
    rf_mat = rfs[:,201-80:201+80].transpose(0,2,1).reshape(rfs.shape[0], -1).copy()
    rf_mat = np.minimum(3.9, rf_mat)
    rf_mat[:,:10] = np.nan
    for k in range(1,rfs.shape[-1]):
        rf_mat[:,k*160-10:k*160+10] = np.nan
    vmax = 5
    im = ax.imshow(rf_mat, aspect="auto", vmin=-vmax, vmax=vmax, cmap=cmap_rb)#"RdBu_r")
    nn = rfs.shape[0]
    for k in range(len(kp_colors)):
        ax.text(k/len(kp_colors),1.005,beh_names[k], color=kp_colors[k], 
                rotation=45, transform=ax.transAxes, size="small")
    ax.set_ylim([0, nn*(1+padding)])
    ax.invert_yaxis()
    ax.axis("off")
    poss = ax.get_position().bounds
    xw = 0.05
    cax = grid.figure.add_axes([poss[0]+poss[2]-xw, poss[1]-poss[3]*0.005, xw, poss[3]*0.01])
    plt.colorbar(im, cax, orientation="horizontal")
    cax.set_xticks([-5,0,5])
    cax.set_xlabel("norm. units")
    return il 

def panels_rasters(fig, grid, il, yh, padding, ipl, sn_test, sn_pred_test, 
                    run, itest, xmin, xmax):
    npl = len(ipl)
    xr = xmax - xmin
    nn = sn_test.shape[0]
    titles = ["spontaneous neural activity (test data)", "behavioral prediction of activity"]
    for k in range(2):
        ax = plt.subplot(grid[5*k + 1 : 5*k + 4])
        pos = ax.get_position().bounds
        ax.axis("off")
        ax.remove()

        # run raster
        padding_x = 0.01
        ax = fig.add_axes([pos[0]+0.02*(k==0), pos[1]+pos[3]*(yh+0.01), pos[2], pos[3]*(1-yh-0.01)])     
        ax.fill_between(np.arange(0, xr), run[itest.flatten()][xmin:xmax], 
                        color=kp_colors[0])
        ax.set_xlim([0*xr, xr*(1+padding_x*2)])
        #ax.set_xlim([0, 1.008*xr])
        ax.set_ylim([0,1.2])
        ax.axis("off")
        transl = mtransforms.ScaledTranslation(-15 / 72, 12 / 72, fig.dpi_scale_trans)
        il = plot_label(ltr, il, ax, transl, fs_title)
        if k==0:
            il+=2

        if k==0:
            ax.text(0.75,0.8,"running speed", transform=ax.transAxes, color=kp_colors[0])
        ax.text(0,1.5,titles[k], transform=ax.transAxes, fontsize="large")
        
        # spk raster
        ax = fig.add_axes([pos[0]+0.02*(k==0), pos[1], pos[2], pos[3]*yh])     
        if k==0:
            ax0 = ax
        xw = pos[2]*0.1
        if k==0:
            cax = fig.add_axes([pos[0]+0.02, pos[1]-pos[3]*0.005, xw, pos[3]*0.01])
        else: 
            cax = None
        plot_raster(ax, sn_test if k==0 else sn_pred_test, 
                    xmin=xmin, xmax=xmax, vmax=1.5, fs=3.38, 
                    nper=50, n_neurons=5000, label=k==1, 
                    padding=padding, padding_x=padding_x, 
                    cax=cax, label_pos="right")    
        if k==0:
            cax = fig.add_axes([pos[0]+0.02-pos[2]*0.02, pos[1], pos[2]*0.01, pos[3]*yh])
            nn = sn_test.shape[0]
            cols = cmap_emb(np.linspace(0, 1, nn))
            cax.imshow(cols[:,np.newaxis], aspect="auto")
            cax.set_ylim([0, (1+padding)*nn])
            cax.invert_yaxis()
            cax.axis("off")
        else:
            for i in range(npl):
                ir = ipl[i]
                xy0 = (0,ir)
                xy1 = (xr,ir)
                con = patches.ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
                                    axesA=ax0, axesB=ax, color=.5*np.ones(3), lw=0.5)
                ax.add_artist(con)
    return il
            
def _fig3(brain_img, face_img, xpos, ypos, isort, 
            isort2, sn, cc_nodes, sn_rand,
            beh, beh_names, tcam, tneural, 
            itest, rfs, run, sn_test, sn_pred_test):
    fig = plt.figure(figsize=(14,7))
    yratio = 14 / 7
    grid = plt.GridSpec(1,9, figure=fig, left=0.02, right=0.98, top=0.94, bottom=0.07, 
                        wspace = 0.35, hspace = 0.3)


    #xmin = 185 
    xmin = 688*4
    xmax = xmin+500
    padding = 0.015
    yh = 0.94 # fraction raster vs run

    npl = 18
    nn = rfs.shape[0]
    ipl = np.linspace(8, nn-8, npl).astype("int")
    print(ipl)

    ax = fig.add_subplot(grid[0])
    ax.axis("off")
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=ax, 
                                                        wspace=0.2, hspace=0.5)
    ax.remove()
    il = 0

    il+=2
    il = panels_beh_traces(grid1, il, face_img, beh, beh_names, tcam, tneural, itest, xmin, xmax)

    il-=3
    il = panel_neuron_pos(fig, grid1, il, yratio, xpos, ypos, isort, brain_img)
    
    il+=2
    il = panels_rfs(grid, il, yh, padding, ipl, rfs, beh_names)
        
    il-=3
    il = panels_rasters(fig, grid, il, yh, padding, ipl, sn_test, sn_pred_test, 
                        run, itest, xmin, xmax)
    return fig

def fig3(root, save_figure=True):
    d = np.load(os.path.join(root, "results", "spont_proc.npz"), allow_pickle=True) 
    try:
        brain_img = plt.imread(os.path.join(root, "figures", "brain_windows.png"))
        face_img = plt.imread(os.path.join(root, "figures", "mouse_face_labeled.png"))
    except:
        brain_img = np.zeros((50,50))
        face_img = np.zeros((50,50))

    fig = _fig3(brain_img, face_img, **d);
    if save_figure:
        fig.savefig(os.path.join(root, "figures", "fig3.pdf"), dpi=200)

def suppfig_random(root, save_figure=True):
    d = np.load(os.path.join(root, "results", "spont_proc.npz"), allow_pickle=True) 
    sn = d["sn"]
    sn_rand = d["sn_rand"]
    run = d["run"]
    itest = d["itest"]

    fig = plt.figure(figsize=(14,8))

    grid = plt.GridSpec(1,2, figure=fig, left=0.02, right=0.99, top=0.9, bottom=0.13, 
                        wspace = 0.15, hspace = 0.3)
    il = 0

    titles = ["random sorting", 
            "Rastermap sorting"]

    xmin = 688*4
    xmax = xmin+500
    padding = 0.015
    yh = 0.94 # fraction raster vs run
    xr = xmax - xmin

    for k in range(2):
        ax = plt.subplot(grid[k])
        pos = ax.get_position().bounds
        ax.axis("off")
        ax.remove()

        # run raster
        ax = fig.add_axes([pos[0]+0.02*(k==0), pos[1]+pos[3]*(yh+0.01), pos[2], pos[3]*(1-yh-0.01)])     
        ax.fill_between(np.arange(0, xr), run[itest.flatten()][xmin:xmax], 
                        color=kp_colors[0])
        ax.set_xlim([0, 1.008*xr])
        ax.set_ylim([0,1.2])
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        transl = mtransforms.ScaledTranslation(-15 / 72, 12 / 72, fig.dpi_scale_trans)
        il = plot_label(ltr, il, ax, transl, fs_title)
        
        if k==0:
            ax.text(0.75,0.8,"running speed", transform=ax.transAxes, color=kp_colors[0])
        ax.text(0,1.5,titles[k], transform=ax.transAxes, fontsize="large")
        
        # spk raster
        ax = fig.add_axes([pos[0]+0.02*(k==0), pos[1], pos[2], pos[3]*yh])     
        if k==0:
            ax0 = ax
        xw = pos[2]*0.1
        if k==0:
            cax = fig.add_axes([pos[0]+0.02, pos[1]-pos[3]*0.005, xw, pos[3]*0.01])
        else: 
            cax = None
        plot_raster(ax, sn_rand[:,itest.flatten()] if k==0 else sn[:, itest.flatten()], 
                    xmin=xmin, xmax=xmax, vmax=1.5, fs=3.38, label=k==0, 
                    nper=50, n_neurons=5000,
                    padding=padding, padding_x=0.01, cax=cax, label_pos="right")    
    
    if save_figure:
        fig.savefig(os.path.join(root, "figures", "suppfig_random.pdf"))


def suppfig_locality(root, save_figure=True):
    d = np.load(os.path.join(root, "results", "asym_vr_spont.npz"), allow_pickle=True) 
    ccs_vr = d["ccs_vr"]
    ccs_spont = d["ccs_spont"]
    localities = d["localities"]
    scores_vr = d["scores_vr"]
    scores_spont = d["scores_spont"]

    fig = plt.figure(figsize=(14,6))
    grid = plt.GridSpec(2,5, figure=fig, left=0.01, right=0.99, top=0.86, bottom=0.01, 
                                wspace = 0.2, hspace = 0.45)
    transl = mtransforms.ScaledTranslation(-15 / 72, 38 / 72, fig.dpi_scale_trans)
    il = 0

    xmin = 185 
    xmax = xmin+500
    xr = xmax - xmin
    yh = 0.9
    padding_x = 0.008
    nbin = 200

    vmax = 0.75
    tstrs = ["neural activity in virtual reality", "spontaneous neural activity"]
    for j in range(2):
        ccs = ccs_vr.copy() if j==0 else ccs_spont.copy()
        scores = scores_vr.copy() if j==0 else scores_spont.copy()
        for k in range(5):
            ax = plt.subplot(grid[j, k])
            cc = ccs[k].copy()
            cc -= np.diag(np.diag(cc))
            im = ax.imshow(cc, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            ax.axis("off")
            
            if k==0:
                ax.text(0., 1.05, f"locality = {localities[k]}\nscores (global / local): {scores[k][0]:.2f} / {scores[k][1]:.2f}", transform=ax.transAxes,
                    fontsize="medium")#, ha="center")
                il = plot_label(ltr, il, ax, transl, fs_title)
                ax.set_title(f"{tstrs[j]} - asymmetric similarity matrix", 
                            y=1.22)
                cax = ax.inset_axes([1.05, 0.55, 0.05, 0.3])
                plt.colorbar(im, cax=cax)
            else:
                ax.text(0., 1.05, f"locality = {localities[k]}\nscores: {scores[k][0]:.2f} / {scores[k][1]:.2f}", transform=ax.transAxes,
                    fontsize="medium")#, ha="center")

    if save_figure:
        fig.savefig(os.path.join(root, "figures", "suppfig_asym.pdf"))

def suppfig_beh(root, save_figure=True):
    d = np.load(os.path.join(root, "results", "spont_proc.npz"), allow_pickle=True) 
    sn = d["sn"]
    itest = d["itest"]
    dbeh = np.load(os.path.join(root, "results", "spont_corrs_beh.npz"), allow_pickle=True) 
    sn_beh = dbeh["sn_beh"]
    vars = dbeh["vars"]

    fig = plt.figure(figsize=(14,3.7))
    grid = plt.GridSpec(1,5, figure=fig, left=0.02, right=0.99, top=0.82, bottom=0.04, 
                                wspace = 0.2, hspace = 0.5)
    transl = mtransforms.ScaledTranslation(-15 / 72, 20 / 72, fig.dpi_scale_trans)
    il = 0

    xmin = 688*4
    xmax = xmin+500
    xr = xmax - xmin
    yh = 0.9
    padding_x = 0.008
    nbin = 50

    padding = 0.015
    ax = plt.subplot(grid[0, 0])
    pos = ax.get_position().bounds
    ax.axis("off")
    ax.remove()
    ax = fig.add_axes([pos[0], pos[1]+pos[3]*(yh+0.03), pos[2], pos[3]*(1-yh-0.01)])     
    ax.text(0, 1.4, "Spontaneous neural activity\nRastermap sorting",
                    fontsize="large", transform=ax.transAxes, fontstyle="italic")
    il = plot_label(ltr, il, ax, transl, fs_title)        
    ax.axis("off")

    ax = fig.add_axes([pos[0], pos[1], pos[2], pos[3]*yh])     
    plot_raster(ax, sn[:,itest.flatten()], 
                    xmin=xmin, xmax=xmax, vmax=1.5, fs=3.38, 
                    nper=nbin, n_neurons=5000, label=True, 
                    padding=padding, padding_x=padding_x)
    
    titles = ["running speed", "whisking speed", "nose speed", "eye area"]
    vmax = [0.2, 0.1, 0.1, 0.2]
    ylim = [[0, 1.2], [0, 3], [0, 3], [-1.5, 3]]
    for k in range(4):
        ax = plt.subplot(grid[0, k+1])
        pos = ax.get_position().bounds
        ax.axis("off")
        ax.remove()

        # beh plot
        ax = fig.add_axes([pos[0], pos[1]+pos[3]*(yh+0.03), pos[2], pos[3]*(1-yh-0.01)])     
        if k<3:
            ax.fill_between(np.arange(0, xr), vars[itest.flatten(),k][xmin:xmax], 
                            color=kp_colors[[0,2,4]][k])
            ax.set_ylim(ylim[k])
        else:
            ax.plot(np.arange(0, xr), vars[itest.flatten(),k][xmin:xmax], color=kp_colors[1])
        ax.set_xlim([-2*padding_x*xr, xr])
        ax.axis("off")
        ax.set_title(titles[k], color=kp_colors[[0,2,4,1]][k], fontsize="medium", y=0.8)
        if k==0:
            il = plot_label(ltr, il, ax, transl, fs_title)        
        elif k==1:
            ax.text(1, 1.8, "sorting by correlation with behavioral variables", ha="center",
                    fontsize="large", transform=ax.transAxes, fontstyle="italic")

        # spk raster
        sn_test = sn_beh[k][::-1]
        #isort = isorts_beh[k][::-1].copy()
        #sn_test = zscore(utils.bin1d(spks[isort][:,itest.flatten()], nbin, axis=0), axis=1)
        ax = fig.add_axes([pos[0], pos[1], pos[2], pos[3]*yh])     
        plot_raster(ax, sn_test, 
                    xmin=xmin, xmax=xmax, vmax=1.5, fs=3.38, 
                    nper=nbin, n_neurons=5000, label=False, 
                    padding=padding, padding_x=padding_x)
        
    if save_figure:
        fig.savefig(os.path.join(root, "figures/suppfig_beh.pdf"), dpi=200)