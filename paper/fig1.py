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
alg_cols = alg_cols[[1,0,2,3]]
alg_cols = np.concatenate((np.array([0,0,0,1])[np.newaxis,:],
                           alg_cols), axis=0)
alg_names = ["rastermap", "t-SNE", "UMAP", "isomap", "laplacian\neigenmaps"]
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
    c0, c1 = 5, 7
    ax_crosscorr.plot(tshifts, cc_tdelay[c0,c1], color=[0.5,0.5,0.5], zorder=1)
    ax_crosscorr.set_ylabel("corr")
    ax_crosscorr.set_xlabel("time lag ($\delta$t)")
    ax_crosscorr.set_xlim([tshifts.min(), tshifts.max()])
    ax_crosscorr.set_title(f"cross-corr\nclusters {c0}, {c1}")
    ix = cc_tdelay[c0,c1].argmax()
    ax_crosscorr.scatter(tshifts[ix], cc_tdelay[c0,c1,ix], marker="*", lw=0.5, color=[1,0.5,0], s=40, zorder=2)
    ax_crosscorr.text(tshifts[ix]+5, cc_tdelay[c0,c1,ix], "max", color=[1,0.5,0], va="center")
    ax_crosscorr.set_ylim([-0.2,0.9])

    nshow=20
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
                        vmin=-1, vmax=1, cmap="RdBu_r")
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
        elif i==2:
            axi.text(-.15,0.5,"clusters", transform=axi.transAxes, rotation=90, va="center")
            axi.text(0.5,-0.15,"clusters", transform=axi.transAxes, ha="center")

    ax = plt.subplot(grid[0,2])
    il = plot_label(ltr, il, ax, transl, fs_title)
    ax.set_title("matching matrix")
    ax.axis("off")
    pos = ax.get_position().bounds
    xi = np.arange(0, nshow, 1, "float32") / nshow
    BBt_log = compute_BBt(xi, xi)
    BBt_log = np.triu(BBt_log)
    BBt_log /= BBt_log.sum()
    BBt_travel = compute_BBt_mask(xi, xi)
    BBt_travel = np.triu(BBt_travel)
    BBt_travel /= BBt_travel.sum() 
    vmax = 2e-2
    for i,mat in enumerate([BBt_log, BBt_travel]):
        axi = fig.add_axes([pos[0]+dx*0.2+i*dx*1, 
                            pos[1]+pos[3]*0.3, pos[2]*0.5, pos[3]*0.5])
        axi.imshow(mat[:nshow,:nshow], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        axi.set_yticks([])
        axi.set_xticks([])
        axi.spines["right"].set_visible(True)
        axi.spines["top"].set_visible(True)
        axi.set_title(["global", "local"][i])
        if i==0:
            axi.text(1.07,0.5, "+",transform=axi.transAxes, fontsize=default_font+6)
            axi.text(1.3,0.5, "w",transform=axi.transAxes, fontsize=default_font+2)
            axi.text(-0.05,0.5, "(1-w)",transform=axi.transAxes, 
                        fontsize=default_font+2, ha="right")
        else:
            axi.text(1.2,0.4, "=",transform=axi.transAxes, fontsize=default_font+6)

    ax = plt.subplot(grid[0,3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+dx*0.2, pos[1]+pos[3]*0.3, pos[2]*0.5, pos[3]*0.5])
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

    ax = plt.subplot(grid[0,4])
    il = plot_label(ltr, il, ax, transl, fs_title)
    for i in range(3):
        nshow=20
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
    ax.set_ylim([-du*0.8,du*2.4])
    ax.axis("off")
    ax.text(0,0.1, "clusters sorted x", transform=ax.transAxes)
    ax.text(0,0., "upsampled nodes -", transform=ax.transAxes)
    ax.text(-0.1,0.5, "weights", rotation=90, transform=ax.transAxes, va="center")
    ax.set_title("upsampling")

    return il

def panels_raster(fig, grid, il, yratio, X_embs, cc_embs, div_map, mod_names, emb_cols):
    transl = mtransforms.ScaledTranslation(-18 / 72, 7 / 72, fig.dpi_scale_trans)
    
    titles = [" simulated neurons sorted by rastermap", " simulated neurons sorted by t-SNE"]
    mod_names_sort = np.array(mod_names.copy())[np.array([3, 4, 0, 2, 1])]#[2,0,3,4,1])]
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
        if k==0: 
            # create bar with colors 
            cax = fig.add_axes([pos[0]+pos[2]*1.01, pos[1], pos[2]*0.01, pos[3]])
            nn = X_embs[k].shape[0]
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
    mod_names_sort = np.array(mod_names.copy())[np.array([3, 4, 0, 2, 1])]
    ax = plt.subplot(grid[1:3, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.02, pos[1]+pos[3]*0.1, pos[2]+0.03, pos[3]*0.82])
    ax.axis("off")
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=ax, 
                                                        wspace=0.1, hspace=0.6)
    ax.remove()

    ids = [3, 2, 0, 4]#[0, 1, 2, 4]
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

def panels_embs(grid, transl, il, xi_all, embs_all, alg_cols, mod_names):
    ax = plt.subplot(grid[3, :3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-0.0, pos[1]-0.02, pos[2]-0.02, pos[3]])
    ax.remove()
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=ax, 
                                                    wspace=0.11, hspace=0.2)
    xip = xi_all.copy()
    embs = embs_all[0].squeeze().copy()
    xip = metrics.emb_to_idx(xip)
    ax = plt.subplot(grid1[0])
    il = plot_label(ltr, il, ax, transl, fs_title)
    ht=ax.text(0, 3000, "ground-truth", va="center")
    ht.set_rotation(90)
    ax.text(0, 7000, "benchmarking embedding algorithms", fontsize="large")
    for k in range(5):
        ax.text(len(xip), 6000 - (1000*k + 500 + 500*(k>3)), mod_names[k], ha="right")
        #ax.plot([0, len(xip)], (i+1) * 1000 * np.ones(2), "--", color="k")
    #ax.axis("square")
    ax.set_ylim([0, len(xip)])
    ax.set_xlim([0, len(xip)])
    ax.axis("off")
    
    subsample = 5
    for k in range(5):
        ax = plt.subplot(grid1[k+1])
        idx = metrics.emb_to_idx(embs[k])
        ax.scatter(5999 - idx[::subsample], 5999 - xip[::subsample], s=1, alpha=0.15, 
                    color=alg_cols[k], rasterized=True)
        for i in range(4):
            ax.plot([0, len(xip)], 6000 - ((i+1) * 1000 * np.ones(2)), "--", color="k", lw=0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)
        ax.set_title(alg_names[k], color=alg_cols[k], fontsize="medium")
        ax.set_ylim([0, len(xip)])
        ax.set_xlim([0, len(xip)])
        if k==2:
            ax.text(0.5, -0.15, "embedding position", transform=ax.transAxes, ha="center")

    return il

def panels_scores(grid, transl, il, scores_all, alg_cols, mod_names):
    ts = 100 * scores_all[:,1].mean(axis=0)
    ts_sem = 100 * scores_all[:,1].std(axis=0) / (scores_all.shape[0]-1)**0.5
    cs = 100 * scores_all[:,0].mean(axis=0)
    cs_sem = 100 * scores_all[:,0].std(axis=0) / (scores_all.shape[0]-1)**0.5

    ax = plt.subplot(grid[3,3])
    il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(5):
        ax.errorbar(np.arange(5), ts[k], ts_sem[k], lw=2,
                    color=alg_cols[k], zorder=0 if k>0 else 5)
    ax.plot(100 * np.ones(5) / 3., color="k", linestyle="--")
    ax.text(0, 33, "chance", va="top")
    ax.set_ylabel("% correct triplets")
    ax.set_ylim([28, 83])
    ax.set_xticks(np.arange(0, 5))
    ax.set_xticklabels(mod_names, 
                        rotation=30, ha="right", rotation_mode="anchor")

    ax = plt.subplot(grid[3,4])
    il = plot_label(ltr, il, ax, transl, fs_title)
    for k in range(5):
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
    div_map = [[5, 42], [43, 108], [109, 136], [136, 170], [174, 199]]
    
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
    ax.set_ylim([10, 65])
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
    ax.set_ylim([28, 83])
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
        ax.text(-0.3, 0.9-0.12*k, l, transform=ax.transAxes,
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