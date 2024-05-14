"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import matplotlib.pyplot as plt 
import os
import numpy as np
from rastermap.utils import bin1d
from rastermap.sort import compute_BBt, compute_BBt_mask
import metrics
from fig_utils import *

alg_cols = plt.get_cmap("tab20b")(np.linspace(0,1.,20))[::4]
alg_cols = alg_cols[[1,0,2,3,4]]
alg_cols = np.concatenate((np.array([0,0,0,1])[np.newaxis,:],
                           alg_cols,
                           np.array([0.5, 0.5, 0.5, 1])[np.newaxis,:]), axis=0)
alg_names = ["rastermap", "t-SNE", "UMAP", "isomap", "laplacian\neigenmaps", "hierarchical\nclustering", "PCA"]
mod_names = ["tuning", "sustained", "sequence 1", "sequence 2", "power-law"]

def panels_schematic(fig, grid, il, cc_tdelay, tshifts, BBt_log, BBt_travel, 
                     U_nodes, U_upsampled, kmeans_img):
    dx = 0.1
    dy = 0.1
    xpad = 0.96 / 5
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    
    ### schematics
    ax_kmeans = plt.subplot(grid[0,0])
    ax_kmeans.axis("off")
    il = plot_label(ltr, il, ax_kmeans, transl, fs_title)
    pos = ax_kmeans.get_position().bounds
    x0 = pos[0]-(xpad-dx)/4
    #print(x0)
    ax_kmeans_img = fig.add_axes([x0+pos[2]*0.15, pos[1]+0.55*pos[3], pos[2]*0.3, pos[3]*0.3])
    ax_crosscorr = fig.add_axes([pos[0]+0.6*pos[2], pos[1]+0.25*pos[3], pos[2]*0.3, pos[3]*0.3])

    # plot kmeans illustration
    ax_kmeans_img.imshow(kmeans_img)
    ax_kmeans_img.set_title("k-means\nclustering")
    ax_kmeans_img.axis("off")

    # plot example crosscorr
    c0, c1 = 12, 14
    ax_crosscorr.plot(tshifts, cc_tdelay[c0,c1], color=[0.5,0.5,0.5], zorder=1)
    ax_crosscorr.set_ylabel("corr")
    ax_crosscorr.set_xlabel("time lag ($\delta$t)")
    ax_crosscorr.set_xlim([tshifts.min(), tshifts.max()])
    ax_crosscorr.set_title(f"cross-corr\nclusters {c0}, {c1}")
    ix = cc_tdelay[c0,c1].argmax()
    ax_crosscorr.scatter(tshifts[ix], cc_tdelay[c0,c1,ix], marker="*", lw=0.5, color=[1,0.5,0], s=40, zorder=2)
    ax_crosscorr.text(tshifts[ix]+5, cc_tdelay[c0,c1,ix], "max", color=[1,0.5,0], va="center")
    ax_crosscorr.set_ylim([-0.2,0.9])

    nshow = 40 #cc_tdelay.shape[0]
    c0 = 0
    ax = plt.subplot(grid[0,1]) 
    il = plot_label(ltr, il, ax, transl, fs_title)
    pos = ax.get_position().bounds
    ax.axis("off")
    ax.set_title("time-lagged correlations")
    pos = ax.get_position().bounds
    dym=0.02
    dxm = dym
    np.random.seed(2)
    isort = np.random.permutation(nshow)
    for i in range(3):
        axi = fig.add_axes([pos[0]+(2-i)*dxm-0.1*pos[2], pos[1]+(-i)*dym+0.3*pos[3], pos[2]*0.7, pos[3]*0.7])
        im=axi.imshow(cc_tdelay[c0:c0+nshow, c0:c0+nshow,11+(2-i)][np.ix_(isort, isort)],#[:nshow,:nshow], 
                        vmin=-0.5, vmax=0.5 , cmap="RdBu_r")
        axi.set_yticks([])
        axi.set_xticks([])
        axi.spines["right"].set_visible(True)
        axi.spines["top"].set_visible(True)
        axi.text(1.05, -0.0, f"$\delta$t = {2-i}", transform=axi.transAxes, ha="left")
        if i==0:
            posi = axi.get_position().bounds
            #divider = make_axes_locatable(ax)
            cax = fig.add_axes([posi[0]+posi[2]+0.005, posi[1]+posi[3]*0.75, posi[2]*0.05, posi[3]*0.25])
            plt.colorbar(im, cax)
            axi.text(1.2, 0.2, "...", transform=axi.transAxes, ha="left", rotation=45, fontweight="bold")
            axi.text(0.9, -0.55, r"$\overline{\text{compute max over } \delta\text{t}}$", transform=axi.transAxes, 
                     ha="left", rotation=42, 
                     fontweight="bold", fontstyle="italic", fontsize="x-small")
            
        elif i==2:
            axi.text(-.15,0.5,"clusters", transform=axi.transAxes, rotation=90, va="center")
            axi.text(0.5,-0.15,"clusters", transform=axi.transAxes, ha="center")

    nshow = 40
    ax = plt.subplot(grid[0,2])
    pos = ax.get_position().bounds
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.set_title("matching matrix")
    ax.axis("off")
    pos = ax.get_position().bounds
    xi = np.arange(0, nshow, 1, "float32") / nshow
    BBt_log = compute_BBt(xi, xi)
    BBt_log /= BBt_log.mean()
    BBt_log = np.triu(BBt_log)
    BBt_travel = compute_BBt_mask(xi, xi)
    BBt_travel /= BBt_travel.mean() 
    BBt_travel = np.triu(BBt_travel)
    vmax = 5
    for i,mat in enumerate([BBt_log, BBt_travel]):
        axi = fig.add_axes([pos[0]+dx*0.2+i*dx*1, 
                            pos[1]+pos[3]*0.3+0.02, pos[2]*0.5, pos[3]*0.5])
        axi.imshow(mat[:nshow,:nshow], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        axi.set_yticks([])
        axi.set_xticks([])
        axi.spines["right"].set_visible(True)
        axi.spines["top"].set_visible(True)
        axi.set_title(["global", "local"][i], fontsize="medium")
        if i==0:
            axi.text(1.07,0.5, "+",transform=axi.transAxes, fontsize=default_font+6)
            axi.text(1.3,0.5, "w",transform=axi.transAxes, fontsize=default_font+2)
            axi.text(-0.05,0.5, "(1-w)",transform=axi.transAxes, 
                        fontsize=default_font+2, ha="right")
        else:
            axi.text(1.2,0.4, "=",transform=axi.transAxes, fontsize=default_font+6)

    ax = fig.add_axes([pos[0]-0.04, pos[1]+pos[3]*0.13, pos[2]*1.71, 0.02])
    arrowprops = dict(arrowstyle="->")
    ax.annotate("", [1, 0], [0, 0], arrowprops=arrowprops)
    ax.axis("off")

    ax = plt.subplot(grid[0,3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+dx*0.2, pos[1]+pos[3]*0.3+0.02, pos[2]*0.5, pos[3]*0.5])
    im = ax.imshow(BBt_log[:nshow, :nshow]/2 + BBt_travel[:nshow, :nshow]/2, 
                    vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    posi = ax.get_position().bounds
    #divider = make_axes_locatable(ax)
    cax = fig.add_axes([posi[0]+posi[2]*1.1, posi[1]+posi[3]*0.6, posi[2]*0.07, posi[3]*0.4])
    plt.colorbar(im, cax)

    pos = ax.get_position().bounds
    ax = fig.add_axes([pos[0]+pos[2]*0.45, pos[1]-0.04, pos[2]*0.1, 0.035])
    ax.annotate("", [0, 0], [0, 1], arrowprops=arrowprops)
    ax.axis("off")
    ax.text(0.5, -0.4, "sorted to match", transform=ax.transAxes, ha="center",
            fontstyle="italic", bbox=dict(facecolor=0.8*np.ones(3), edgecolor="none"))

    ax = fig.add_axes([pos[0]+pos[2]*0.45, pos[1]-0.095, pos[2]*0.1, 0.035])
    ax.annotate("", [0, 0], [0, 1], arrowprops=arrowprops)
    ax.axis("off")

    ax = plt.subplot(grid[0,4])
    il = plot_label(ltr, il, ax, transl, fs_title)
    for i in range(3):
        nshow=20
        c0 = 10
        du = 0.4
        p = ax.plot(np.linspace(0, len(U_upsampled[c0:c0+nshow*10,i]), 
                                            len(U_nodes[c0:c0+nshow+1,i])), 
                            U_nodes[c0:c0+nshow+1,i] + du*(2-i) + (i==2)*0.08, "x", 
                            markersize=6, 
                            color=np.ones(3)*(0.25*i),
                            lw=4)
        ax.plot(U_upsampled[c0*10:(c0+nshow)*10,i] + du*(2-i) + (i==2)*0.08, 
                color=p[0].get_color(), lw=1)
        ax.text(20*10, du*(2-i) + du*0.05 + U_nodes[c0:c0+nshow,i].max(), f"PC {i+1:d}", 
                         color=p[0].get_color(), ha="right")
    ax.set_ylim([-du*0.5,du*2.])
    ax.axis("off")
    ax.text(0,0.1, "clusters sorted x", transform=ax.transAxes)
    ax.text(0,0., "upsampled nodes -", transform=ax.transAxes)
    ax.text(-0.1,0.5, "weights", rotation=90, transform=ax.transAxes, va="center")
    ax.set_title("upsampling")

    return il

def panels_raster(fig, grid, il, yratio, X_embs, cc_embs, div_map=None, mod_names=None, emb_cols=None):
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    
    titles = [" simulated neurons sorted by rastermap", " simulated neurons sorted by t-SNE"]
    nn = X_embs[0].shape[0]
    for k in range(2):
        ax = plt.subplot(grid[k+1,0:3])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1], pos[2]*0.94, pos[3]])
        pos = ax.get_position().bounds
        if k==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            cax = fig.add_axes([pos[0]+pos[2]-pos[3]*0.25, pos[1]-pos[3]*0.05, pos[3]*0.25, pos[2]*0.01])
        else:
            cax = None
        plot_raster(ax, X_embs[k], xmin=0, xmax=8000, label=1-k, cax=cax)
        if cax is not None:
            cax.set_xlabel("")
            cax.text(-0.15,-2, "z-scored", transform=cax.transAxes, ha="right")
        #ax.text(0.05, 1.02, titles[k], transform=ax.transAxes, ha="left", fontsize="large")
        ax.set_title(titles[k])
        if k==0 and mod_names is not None: 
            mod_names_sort = np.array(mod_names.copy())[np.array([1, 2, 3, 0, 4])]
            # create bar with colors 
            cax = fig.add_axes([pos[0]+pos[2]*1.01, pos[1], pos[2]*0.01, pos[3]])
            cax.imshow(emb_cols[:,np.newaxis], aspect="auto")
            cax.set_ylim([0, nn*1.025])
            cax.invert_yaxis()
            cax.axis("off")

            # create bar with ticks 
            cax = fig.add_axes([pos[0]+pos[2]*1.03, pos[1], pos[2]*0.01, pos[3]])
            for d in range(len(div_map)):
                cax.plot([0,0], [div_map[d][0], div_map[d][1]], marker=0, color="k")
                cax.text(0.08, (div_map[d][1]-div_map[d][0])/2 + div_map[d][0], 
                         mod_names_sort[d], va="center")
            cax.set_ylim([0, nn*1.025])
            cax.invert_yaxis()
            cax.axis("off")

        ax = plt.subplot(grid[k+1,3])
        pos2 = ax.get_position().bounds
        ax.set_position([pos2[0], pos2[1], pos[3]/yratio, pos[3]])
        vmax = 0.3
        cc = cc_embs[k].copy()
        im = ax.imshow(cc, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim([0, nn*1.025])
        posi = ax.get_position().bounds
        if k==0:
            ax.set_title("asymmetric similarity")

            il = plot_label(ltr, il, ax, transl, fs_title)
            cax = fig.add_axes([posi[0], posi[1]-posi[3]*0.05, posi[3]*0.25, posi[2]*0.05])
            plt.colorbar(im, cax, orientation="horizontal")
            cax.set_xticks([-0.3,0,0.3])
        ax.set_ylim([0, nn*1.025])
        ax.set_xlim([0, nn*1.025])
        ax.invert_yaxis()
        ax.axis("off")

    return il

def panels_responses(grid, transl, il, div_map, seqcurves0, seqcurves1, tcurves, xresp, 
                     emb_cols, mod_names):
    mod_names_sort = np.array(mod_names.copy())[np.array([1, 2, 3, 0, 4])]
    ax = plt.subplot(grid[1:3, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.02, pos[1]+pos[3]*0.1, pos[2]+0.03, pos[3]*0.82])
    ax.axis("off")
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=ax, 
                                                        wspace=0.1, hspace=0.6)
    ax.remove()

    #ids = [3, 2, 0, 4]#[0, 1, 2, 4]
    ids = [1, 3, 2, 0]
    nsp = [4, 3, 4, 4]
    xlabels = ["position", "stim id", "position", "time (sec.)"]
    for a in range(4):
        ax = plt.subplot(grid1[a//2,a%2])
        if a==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
        dt = ids[a]
        n_x = div_map[dt][1] - div_map[dt][0]
        if a==0 or a==2:
            tc = seqcurves0.copy() if a==0 else seqcurves1.copy()
            tc = tc[div_map[dt][0] : div_map[dt][1]]
            ax.set_xticks([0,25,50])
        elif a==1:
            tc = tcurves[div_map[dt][0] : div_map[dt][1]].copy()
            ax.set_xticks(np.arange(0,16,5))
        elif a==3:
            tc = xresp[div_map[dt][0] : div_map[dt][1]].copy()
            tc = bin1d(tc, 10, axis=1)
            ax.set_xticks([0, 20])
            ax.set_xticklabels(["0", "10"])
        tc -= tc.min(axis=0)
        if a < 3:
            tc /= tc.max(axis=0)
        for i, c in enumerate(tc[(a<3)::nsp[a]]):
            ax.plot(c, lw=1, color=emb_cols[div_map[dt][0] + i*nsp[a] + (a<3)]);
        ax.set_xlabel(xlabels[a])
        ax.set_title(mod_names_sort[dt], fontsize="medium")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        if a==0: 
            ax.text(1., 1.3, "superneuron responses", transform=ax.transAxes, 
                    ha="center", fontsize="large")
    return il 

def panels_embs(grid, transl, il, xi_all, embs_all, alg_cols, mod_names=None, y=3,
                title="benchmarking embedding algorithms"):
    ax = plt.subplot(grid[y, :3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.01, pos[1]-0.02, pos[2]-0.0, pos[3]])
    ax.remove()
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 7 + (mod_names is not None), subplot_spec=ax, 
                                                    wspace=0.11, hspace=0.2)
    embs = embs_all[0].squeeze().copy() if embs_all.squeeze().ndim > 2 else embs_all.squeeze()
    if mod_names is not None:
        ax = plt.subplot(grid1[0])
        il = plot_label(ltr, il, ax, transl, fs_title)
        ax.text(0, 7000, title, fontsize="large")
        ht=ax.text(-1500, 3000, "ground-truth", va="center")
        ht.set_rotation(90)
        xip = metrics.emb_to_idx(xi_all.copy())
        for k in range(5):
            ax.text(len(xip), 6000 - (1000*k + 500 + 500*(k>3)), mod_names[k], ha="right")
        #ax.axis("square")
        ax.set_ylim([0, len(xip)])
        ax.set_xlim([0, len(xip)])
        ax.axis("off")
    else:
        ht=ax.text(5000, 3000, "ground-truth", va="center")
        ht.set_rotation(90)
        xip = len(xi_all) * (xi_all.squeeze().copy())
        #ax.plot([0, len(xip)], (i+1) * 1000 * np.ones(2), "--", color="k")
    
    subsample = 5
    for k in range(7):
        ax = plt.subplot(grid1[k+(mod_names is not None)])
        idx = metrics.emb_to_idx(embs[k])
        ax.scatter(5999 - idx[::subsample], 5999 - xip[::subsample], s=1, alpha=0.15, 
                    color=alg_cols[k], rasterized=True)
        if mod_names is not None:
            for i in range(4):
                ax.plot([0, len(xip)], 6000 - ((i+1) * 1000 * np.ones(2)), "--", color="k", lw=0.5)
        elif k==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            ax.text(0, 7000, title, fontsize="large")
            ht=ax.text(-1500, 3000, "ground-truth", va="center")
            ht.set_rotation(90)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.set_title(alg_names[k], color=alg_cols[k], fontsize="medium")
        ax.set_ylim([0, len(xip)])
        ax.set_xlim([0, len(xip)])
        if k==3:
            ax.text(0.5, -0.15, "embedding position", transform=ax.transAxes, ha="center")
        
    return il

def panels_scores(grid, transl, il, scores_all, alg_cols, mod_names, y=3, x=3):
    ts = 100 * scores_all[:,1].mean(axis=0)
    ts_sem = 100 * scores_all[:,1].std(axis=0) / (scores_all.shape[0]-1)**0.5
    cs = 100 * scores_all[:,0].mean(axis=0)
    cs_sem = 100 * scores_all[:,0].std(axis=0) / (scores_all.shape[0]-1)**0.5

    ax = plt.subplot(grid[y,x])
    il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(7):
        ax.errorbar(np.arange(5), ts[k], ts_sem[k], lw=2,
                    color=alg_cols[k], zorder=0 if k>0 else 5)
    ax.plot(100 * np.ones(5) / 3., color="k", linestyle="--")
    ax.text(0, 33, "chance", va="top")
    ax.set_ylabel("% correct triplets")
    ax.set_ylim([28, 88 if ts.max() < 88 else 91])
    ax.set_xticks(np.arange(0, 5))
    ax.set_xticklabels(mod_names, 
                        rotation=30, ha="right", rotation_mode="anchor")

    ax = plt.subplot(grid[y,x+1])
    il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(7):
        ax.errorbar(np.arange(5), cs[k], cs_sem[k], lw=2, 
                    color=alg_cols[k], zorder=0 if k>0 else 5)
    ax.plot(100 * np.array([5./6, 5./6, 5./6, 5./6, 2./3]), color="k", linestyle="--")
    ax.text(0, 85, "chance")
    ax.set_ylabel("% contamination")
    ax.set_xticks(np.arange(0, 5))
    ax.set_xticklabels(mod_names, 
                        rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim([0, 85])
    return il

def _fig1(kmeans_img, xi_all, cc_tdelay, tshifts, BBt_log, BBt_travel, 
        seqcurves0, seqcurves1, tcurves, xresp, 
        U_nodes, U_upsampled, X_embs, cc_embs,cc_embs_max,
        csig, scores_all, embs_all):
    
    fig = plt.figure(figsize=(14,10))
    yratio = 14 / 10
    grid = plt.GridSpec(4,5, figure=fig, left=0.04, right=0.98, top=0.96, bottom=0.07, 
                        wspace = 0.35, hspace = 0.25)

    
    # divide up plot into modules
    #div_map = [[0, 35], [35, 63], [67, 102], [106, 172], [174, 196]]
    #mod_names = ["tuning", "sustained", "sequence 1", "sequence 2", "power-law"]
    #div_map = [[5, 42], [43, 108], [109, 136], [136, 170], [174, 199]]
    div_map = [[0, 29], [29,64], [65,101], [101, 130], [130,199]]
    
    il = 0
    il = panels_schematic(fig, grid, il, cc_tdelay, tshifts, BBt_log, BBt_travel, 
                         U_nodes, U_upsampled, kmeans_img)

    emb_cols = plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95, X_embs[0].shape[0]))[::-1]
    il = panels_raster(fig, grid, il, yratio, X_embs, cc_embs_max, div_map, mod_names, emb_cols)

    transl = mtransforms.ScaledTranslation(-18 / 72, 30 / 72, fig.dpi_scale_trans)
    il = panels_responses(grid, transl, il, div_map, seqcurves0, seqcurves1, tcurves, xresp, 
                        emb_cols, mod_names)

    transl = mtransforms.ScaledTranslation(-18 / 72, 26 / 72, fig.dpi_scale_trans)
    il = panels_embs(grid, transl, il, xi_all, embs_all, alg_cols, mod_names)

    transl = mtransforms.ScaledTranslation(-40 / 72, 4 / 72, fig.dpi_scale_trans)
    panels_scores(grid, transl, il, scores_all, alg_cols, mod_names)

    return fig 

def fig1(root, save_figure=True):
    d1 = np.load(os.path.join(root, "simulations", "sim_results.npz"), allow_pickle=True) 
    d2 = np.load(os.path.join(root, "simulations", "sim_performance.npz"), allow_pickle=True) 
    try:
        kmeans_img = plt.imread(os.path.join(root, "figures", "manifold_kmeans.png"))
    except:
        kmeans_img = np.zeros((50,50))

    fig = _fig1(kmeans_img, **d1, **d2);
    if save_figure:
        fig.savefig(os.path.join(root, "figures", "fig1.pdf"), dpi=200)            

def panel_mnn(ax, mnn_all, knn, cols):
    ms = 100 * mnn_all.mean(axis=0)
    ms_sem = 100 * mnn_all.std(axis=0) / (mnn_all.shape[0]-1)**0.5
    #il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(len(ms)):
        ax.errorbar(knn/2000*100, ms[k], ms_sem[k], lw=2,
                    color=cols[k], zorder=0 if k>0 else 5)
    ax.set_ylim([0, 68])
    ax.set_ylabel("% neighbors preserved")
    ax.set_xlabel("top n % of neighbors")

def panels_scores_tsne_umap(fig, grid, il, transl, scores_all, mnn_all, knn, 
                            cols, perplexities=None, n_neighbors=None, g0=0):
    legend = ["rastermap"]
    extra = ["", "(default)"]
    if perplexities is not None:
        for p in perplexities:
            if p[1]>0:
                legend.append(f"t-SNE $P$ = [{p[0]}, {p[1]}]")
            else:
                legend.append(f"t-SNE $P$ = {p[0]} {extra[p[0]==30]}")
    elif n_neighbors is not None:
        for nn in n_neighbors:
            legend.append(f"UMAP $nn$ = {nn} {extra[nn==15]}")

    ax = plt.subplot(grid[0, 2+2*g0])
    il = plot_label(ltr, il, ax, transl, fs_title)
    panel_mnn(ax, mnn_all, knn, cols)

    ax = plt.subplot(grid[0, 3+2*g0])
    for k, l in enumerate(legend):
        ax.text(-0.3, 0.9-0.12*k, l, transform=ax.transAxes,
                 ha="left", color=cols[k])
    ax.axis("off")
    
    ts = 100 * scores_all[:,1].mean(axis=0)
    ts_sem = 100 * scores_all[:,1].std(axis=0) / (scores_all.shape[0]-1)**0.5
    ax = plt.subplot(grid[1, 2+2*g0])
    #il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(len(ts)-1):
        ax.errorbar(np.arange(5), ts[k], ts_sem[k], lw=2,
                    color=cols[k], zorder=0 if k>0 else 5)
    ax.plot(100 * np.ones(5) / 3., color="k", linestyle="--")
    ax.text(0, 33, "chance", va="top")
    ax.set_ylabel("% correct triplets")
    ax.set_ylim([28, 88])
    ax.set_xticks(np.arange(0, 5))
    ax.set_xticklabels(mod_names, 
                        rotation=30, ha="right", rotation_mode="anchor")

    cs = 100 * scores_all[:,0].mean(axis=0)
    cs_sem = 100 * scores_all[:,0].std(axis=0) / (scores_all.shape[0]-1)**0.5
    ax = plt.subplot(grid[1, 3+2*g0])
    #il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(len(cs)-1):
        ax.errorbar(np.arange(5), cs[k], cs_sem[k], lw=2, 
                    color=cols[k]   , zorder=0 if k>0 else 5)
    ax.plot(100 * np.array([5./6, 5./6, 5./6, 5./6, 2./3]), color="k", linestyle="--")
    ax.text(0, 85, "chance")
    ax.set_ylabel("% contamination")
    ax.set_ylim([0, 85])
    ax.set_xticks(np.arange(0, 5))
    ax.set_xticklabels(mod_names, 
                        rotation=30, ha="right", rotation_mode="anchor")
    return il
        
def suppfig_scores(root, save_figure=True):
    tsne_cols = plt.get_cmap("Greens")(np.linspace(0.4,1,8))
    tsne_cols = np.concatenate((np.array([0,0,0,1])[np.newaxis,:],
                                tsne_cols), axis=0)
    tsne_cols[2] = alg_cols[1]

    umap_cols = plt.get_cmap("Blues")(np.linspace(0.3,1,8))
    umap_cols = np.concatenate((np.array([0,0,0,1])[np.newaxis,:],
                                umap_cols), axis=0)
    umap_cols[2] = alg_cols[2]

    d2 = np.load(os.path.join(root, "simulations", "sim_performance.npz"), allow_pickle=True) 
    dtsne = np.load(os.path.join(root, "simulations", "sim_performance_tsne.npz"))
    dumap = np.load(os.path.join(root, "simulations", "sim_performance_umap.npz"))
    dneigh = np.load(os.path.join(root, "simulations", "sim_performance_neigh.npz"))

    fig = plt.figure(figsize=(14,5))
    grid = plt.GridSpec(2, 6, figure=fig, left=0.06, right=0.98, top=0.94, bottom=0.15, 
                        wspace = 0.5, hspace = 0.4)
    transl = mtransforms.ScaledTranslation(-45 / 72, 5 / 72, fig.dpi_scale_trans)
    il = 0

    mnn_all = dneigh["mnn_all"]
    knn = dneigh["knn"]
    ax = plt.subplot(grid[0, 0])
    il = plot_label(ltr, il, ax, transl, fs_title)
    panel_mnn(ax, mnn_all, knn, alg_cols)

    ax = plt.subplot(grid[0, 1])
    for k, l in enumerate(alg_names):
        ax.text(-0.3, 0.9-0.12*k, l.replace("\n", " "), transform=ax.transAxes,
                    ha="left", va="top", color=alg_cols[k])
    ax.axis("off")

    scores_rmap = d2["scores_all"][:,:,0]
    mnn_rmap = dneigh["mnn_all"][:,0]
    knn = dneigh["knn"]

    mnn_all = dtsne["mnn_all"]
    mnn_all = np.concatenate((mnn_rmap[:,np.newaxis], mnn_all), axis=1)
    scores_all = dtsne["scores_all"]
    scores_all = np.concatenate((scores_rmap[:,:,np.newaxis], scores_all), axis=2)
    perplexities = dtsne["perplexities"]
    il = panels_scores_tsne_umap(fig, grid, il, transl, scores_all, mnn_all, 
                            knn, tsne_cols, perplexities=perplexities, g0=0)

    mnn_all = dumap["mnn_all"]
    mnn_all = np.concatenate((mnn_rmap[:,np.newaxis], mnn_all), axis=1)
    scores_all = dumap["scores_all"]
    scores_all = np.concatenate((scores_rmap[:,:,np.newaxis], scores_all), axis=2)
    n_neighbors = dumap["n_neighbors"]
    il = panels_scores_tsne_umap(fig, grid, il, transl, scores_all, mnn_all, 
                            knn, umap_cols, n_neighbors=n_neighbors, g0=1)

    if save_figure:
        fig.savefig(os.path.join(root, "figures", "suppfig_scores.pdf"), dpi=200)

def suppfig_spont(root, save_fig=True):
    dat = np.load(os.path.join(root, "simulations", "sim_spont_performance.npz"))
    embs = dat["embs"]
    X_embs = dat["X_embs"]
    cc_embs = dat["cc_embs"]
    corrs_all = dat["corrs_all"]
    xi_all = dat["xi_all"]
    il = 0
    fig = plt.figure(figsize=(14,10))
    yratio = 14 / 10
    grid = plt.GridSpec(4,6, figure=fig, left=0.03, right=0.99, top=0.96, bottom=0.09, 
                        wspace = 0.35, hspace = 0.35)

    ### gray background
    ax = plt.subplot(grid[:2, 3:])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.01, pos[1]-0.01, pos[2]*1.03, pos[3]*1.06])
    pos = ax.get_position().bounds
    ax.spines["left"].set_visible(0)
    ax.spines["bottom"].set_visible(0)
    ax.patch.set_facecolor(0.9 * np.ones(3))
    ax.set_xticks([])
    ax.set_yticks([])
    xx = 0.86
    ax = fig.add_subplot([xx, 0.3, pos[2]+pos[0]-xx, 0.4])
    ax.spines["left"].set_visible(0)
    ax.spines["bottom"].set_visible(0)
    ax.patch.set_facecolor(0.9 * np.ones(3))
    ax.set_xticks([])
    ax.set_yticks([])

    transl = mtransforms.ScaledTranslation(-15 / 72, 7 / 72, fig.dpi_scale_trans)
    titles = [" power-law only simulation, sorted by rastermap", " sorted by t-SNE"]
    nn = X_embs[0].shape[0]
    for k in range(2):
        ax = plt.subplot(grid[k,0:3])
        pos = ax.get_position().bounds
        ax.set_position([pos[0], pos[1], pos[2], pos[3]])
        pos = ax.get_position().bounds
        if k==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            cax = fig.add_axes([pos[0]+pos[2]-pos[3]*0.25, pos[1]-pos[3]*0.05, pos[3]*0.25, pos[2]*0.01])
        else:
            cax = None
        plot_raster(ax, X_embs[k], xmin=0, xmax=8000, label=1-k, cax=cax)
        if cax is not None:
            cax.set_xlabel("")
            cax.text(-0.15,-2, "z-scored", transform=cax.transAxes, ha="right")
        #ax.text(0.05, 1.02, titles[k], transform=ax.transAxes, ha="left", fontsize="large")
        ax.set_title(titles[k])
        

    ky = [0,1,0,1,0,1,2]
    kx = [3,3,4,4,5,5,5]
    transl = mtransforms.ScaledTranslation(-25 / 72, 7 / 72, fig.dpi_scale_trans)
    for k in range(7):
        ax = plt.subplot(grid[ky[k],kx[k]])
        pos2 = ax.get_position().bounds
        ax.set_position([pos2[0], pos2[1], pos[3]/yratio, pos[3]])
        vmax = 0.3
        cc = cc_embs[k].copy()
        im = ax.imshow(cc, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.set_yticks([])
        ax.set_xticks([])
        posi = ax.get_position().bounds
        if k==0:
            il = plot_label(ltr, il, ax, transl, fs_title)
            cax = fig.add_axes([posi[0], posi[1]-posi[3]*0.05, posi[3]*0.25, posi[2]*0.05])
            plt.colorbar(im, cax, orientation="horizontal")
            cax.set_xticks([-0.3,0,0.3])
        elif k==2:
            ax.set_title("Correlation matrices", y=1.07, fontstyle="italic")
        ax.text(0.5, 1.08, alg_names[k].replace("\n", " "), transform=ax.transAxes, ha="center", va="top")
        ax.set_ylim([0, nn*1.025])
        ax.set_xlim([0, nn*1.025])
        ax.invert_yaxis()
        ax.axis("off")

    transl = mtransforms.ScaledTranslation(-12 / 72, 25 / 72, fig.dpi_scale_trans)
    il = panels_embs(grid, transl, il, xi_all, embs, alg_cols, 
                     mod_names=None, y=2, title="embedding sortings, power-law only")

    transl = mtransforms.ScaledTranslation(-60 / 72, 1 / 72, fig.dpi_scale_trans)
    ax = plt.subplot(grid[2, -3:-1])
    il = plot_label(ltr, il, ax, transl, fs_title)
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.01, pos[1]+0.0, pos[2]*1.06, pos[3]])
    for k in range(7):
        ax.scatter(np.ones(10)*k, corrs_all[:,k], s=15, color=alg_cols[k])
        ax.plot(k + np.array([-0.3, 0.3]), corrs_all[:,k].mean() * np.ones(2), 
                color=alg_cols[k], lw=2)
        if k==4 or k==5:
            ax.text(k-0.2, -0.03, alg_names[k], color=alg_cols[k], rotation=30, 
                fontsize="small", va="top", ha="center")
        else:
            ax.text(k, -0.06, alg_names[k], color=alg_cols[k], rotation=0, 
                fontsize="small", va="top", ha="center")
    ax.set_ylabel("correlation w/ ground-truth")
    ax.set_xticks(np.arange(0, 7))
    ax.set_xticklabels([])
    plt.ylim([-0.01, 1.01])

    ax = plt.subplot(grid[-1, :])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.0, pos[1]-pos[3]*0.1, pos[2], pos[3]])
    ax.axis("off")
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1,5, subplot_spec=ax, 
                                                        wspace=0.5, hspace=0.)
    ax.remove()


    dat = np.load(os.path.join(root, "simulations", "sim_no_add_spont_performance.npz"))
    scores_all = dat["scores_all"]
    embs = dat["embs_all"][0]
    xi_all = dat["xi_all"]
    transl = mtransforms.ScaledTranslation(-13 / 72, 25 / 72, fig.dpi_scale_trans)
    il = panels_embs(grid1, transl, il, xi_all, embs, alg_cols, 
                     mod_names=mod_names, y=0, title="embedding sortings, no power-law noise")
    
    transl = mtransforms.ScaledTranslation(-30 / 72, 5 / 72, fig.dpi_scale_trans)
    panels_scores(grid1, transl, il, scores_all, alg_cols, mod_names, y=0)
    

    if save_fig:
        fig.savefig(os.path.join(root, "figures", "suppfig_spont.pdf"), dpi=200)


def suppfig_repro(root, save_fig=True):
    d = np.load(os.path.join(root, "simulations", "sim_0_reproducibility.npz"))
    corrs_all = d["corrs_all"]
    scores_all = d["scores_all"]
    fig = plt.figure(figsize=(8,3))
    transl = mtransforms.ScaledTranslation(-35 / 72, 15 / 72, fig.dpi_scale_trans)
    grid = plt.GridSpec(1,3, figure=fig, left=0.07, right=0.97, top=0.85, bottom=0.22, 
                        wspace = 0.35, hspace = 0.35)
    il = 0
    
    ax = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(2):
        ax.scatter(np.tile(np.arange(5)[:,np.newaxis], (1,20)) + np.random.randn(5,20)*0.05 + (k-0.5)*0.3, 
                    100 * scores_all[1, k*20:(k+1)*20].T, 
                    s=15, color=alg_cols[k])
        #ax.errorbar(np.arange(5), ts[k], ts_sem[k], lw=1,
        #            color=alg_cols[k], zorder=0 if k>0 else 5)
        ax.text(-0.3, 83-k*6, alg_names[k], color=alg_cols[k])
    ax.plot(100 * np.ones(5) / 3., color="k", linestyle="--")
    ax.text(0, 33, "chance", va="top")
    ax.set_ylabel("% correct triplets")
    ax.set_ylim([28, 86])
    ax.set_xticks(np.arange(0, 5))

    ax.set_xticklabels(mod_names, 
                        rotation=30, ha="right", rotation_mode="anchor")
    ax.set_title("Embedding quality across random seeds", y=1.05)

    ax = plt.subplot(grid[0,1])
    for k in range(2):
        ax.scatter(np.tile(np.arange(5)[:,np.newaxis], (1,20)) + np.random.randn(5,20)*0.05 + (k-0.5)*0.3, 
                    100 * scores_all[0, k*20:(k+1)*20].T, 
                    s=15, color=alg_cols[k])
    ax.plot(100 * np.array([5./6, 5./6, 5./6, 5./6, 2./3]), color="k", linestyle="--")
    ax.text(0, 85, "chance")
    ax.set_ylabel("% contamination")
    ax.set_xticks(np.arange(0, 5))
    ax.set_xticklabels(mod_names, 
                        rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim([0, 85])

    ax = plt.subplot(grid[0,2])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+0.01, pos[1], pos[2]-0.04, pos[3]])
    il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(len(corrs_all)):
        ax.scatter(np.ones(20)*k + np.random.randn(20)*0.05, corrs_all[k], 
                    s=10, color=alg_cols[k])
        #ax.plot(k + np.array([-0.3, 0.3]), corrs_all[k].mean() * np.ones(2), 
        #            color=alg_cols[k], lw=2)
        ax.text(k+0.05, -0.05, alg_names[k], color=alg_cols[k], rotation=30,
                    ha="right", va="top")
    ax.set_ylabel("correlation w/ ground-truth")
    ax.set_xticks(np.arange(0, len(corrs_all)))
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0, 1.1,0.2))
    ax.set_ylim([-0.01, 1.01])
    ax.set_title("Power-law module only", y=1.05)

    if save_fig:
        fig.savefig(os.path.join(root, "figures", "suppfig_repro.pdf"), dpi=200)


def suppfig_params(root, save_fig=True):
    d_tl_loc = np.load(os.path.join(root, "simulations", "sim_performance_tl_loc.npz"))
    d_nclust = np.load(os.path.join(root, "simulations", "sim_performance_nclust.npz"))
    d_leiden = np.load(os.path.join(root, "simulations", "sim_performance_leiden.npz"))

    il = 0
    fig = plt.figure(figsize=(10,8))
    yratio = 10 / 8
    grid = plt.GridSpec(4,5, figure=fig, left=0.05, right=0.97, top=0.95, bottom=0.1, 
                        wspace = 0.35, hspace = 0.6)

    vmin = [28, 0]
    vmax = [88, 85]

    transl = mtransforms.ScaledTranslation(-35 / 72, 15 / 72, fig.dpi_scale_trans)

    tstr = ["% correct triplets", "% contamination"]
    scores_leiden = 100 * d_leiden["scores_all"][:,::-1,1]
    nclust_leiden = d_leiden["nclust_leiden"]
    scores_all = 100 * d_nclust["scores_all"][:,::-1,:-1]
    nclust = d_nclust["nclust"]
    for j in range(5):
        for k in range(2):
            ax = plt.subplot(grid[k, j])
            if k==1:
                pos = ax.get_position().bounds
                ax.set_position([pos[0], pos[1]+0.04, pos[2], pos[3]])
            ax.errorbar(nclust, scores_all[:,k,:,j].mean(axis=0), 
                        scores_all[:,k,:,j].std(axis=0) / 9**0.5, color="k")
            ax.scatter(nclust_leiden, scores_leiden[:,k,j], marker="x", 
                        color=0.6*np.ones(3))
            ax.set_yticks(np.arange(0, 90, 20))
            ax.set_ylim([vmin[k], vmax[k]])
            ax.set_xlim([25, 150])
            if j==0:
                if k==0:
                    ax.text(1, 0.9, "k-means", color="k", ha="right", transform=ax.transAxes)
                    ax.text(1, 0.75, "leiden", color=0.6*np.ones(3), ha="right", transform=ax.transAxes)
                    il = plot_label(ltr, il, ax, transl, fs_title)
                    ax.text(30, 35, "chance", va="bottom", fontsize="small")
                else:
                    ax.text(30, 82, "chance", va="top", fontsize="small")
                    ax.set_xlabel("number of clusters")
                ax.set_ylabel(tstr[k])
            if k==0:
                ax.plot(np.array([0, 200]), 100 * np.ones(2) / 3., color="k", linestyle="--")
                ax.set_title(mod_names[j], loc="center", y=1.1)
            else:
                ax.plot(np.array([0, 200]), 100 * np.ones(2) * 5./6 if j<4 else 100 * np.ones(2) * 2./3, color="k", linestyle="--")
        

    tl = d_tl_loc["tl"]
    loc = d_tl_loc["loc"]
    scores_all = 100 * d_tl_loc["scores_all"][:,::-1,:-1].reshape(*d_tl_loc["scores_all"].shape[:2],6,7,-1)

            
    for j in range(5):
        for k in range(2):
            ax = plt.subplot(grid[k+2, j])
            if k==1:
                pos = ax.get_position().bounds
                ax.set_position([pos[0], pos[1]-0.03, pos[2], pos[3]])
            im = ax.imshow(scores_all[:,k,:,:,j].mean(axis=0), 
                        vmin=vmin[k], vmax=vmax[k])
            ax.set_xticks(np.arange(0,7))
            ax.set_yticks(np.arange(0,6))
            ax.set_xticklabels([str(t) for t in np.unique(loc)], rotation=90)
            ax.set_yticklabels([str(t) for t in np.unique(tl)])
            if j==0:
                ax.text(1.55, 1.3, tstr[k], transform=ax.transAxes, ha="center")
                cax = ax.inset_axes([1.4, 1.2, 0.3, 0.05])
                plt.colorbar(im, cax=cax, orientation="horizontal")
                ax.set_xlabel("locality")
                ax.set_ylabel("time_lag_window")
                if k==0:
                    il = plot_label(ltr, il, ax, transl, fs_title)

    if save_fig:
        fig.savefig(os.path.join(root, "figures", "suppfig_params.pdf"), dpi=200)
            