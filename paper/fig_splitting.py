"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
from matplotlib import patches, collections
from scipy.io import loadmat
from fig_utils import *


def example_basis_fcns():
    kx,ky = np.meshgrid(np.arange(0,31,1,"float32"), np.arange(0,31,1,"float32"))
    kx = kx.flatten()
    ky = ky.flatten()
    kx = kx[1:]
    ky = ky[1:]
    xx, yy = np.meshgrid(np.arange(0,1,0.02,"float32"), np.arange(0,1,0.02,"float32"))
    xx = xx.flatten()[:,np.newaxis]
    yy = yy.flatten()[:,np.newaxis]
    dk = (kx**2 + ky**2)**0.5
    kx = kx[dk.argsort()]
    ky = ky[dk.argsort()]
    B = np.cos(np.pi * xx * kx ) * np.cos(np.pi * yy * ky )
    return B

def plot_sorting2d(ax, xi, isort, label=False):
    xi_sort = xi[isort][:,np.newaxis,:][::2]
    segments = np.concatenate([xi_sort[:-1], xi_sort[1:]], axis=1)
    #segments = segments[::5]
    cmap = cmap_emb(np.linspace(0.1,1.0,segments.shape[0]))[:,:3]
    lc = collections.LineCollection(segments, colors=cmap, alpha=1)
    lc.set_linewidth(0.5)
    line = ax.add_collection(lc)
    if label:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_facecolor("k")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.axis("square")
    return line

def panels_sim2d(fig, grid, il, xi, X_embedding, isorts, 
                 knn_score, knn, rhos):
    perplexities = [[10, 100], [10], [30], [100], [300]]

    gs = plt.get_cmap("Greens")(np.linspace(0.3, 0.7, knn_score.shape[1]-1))[::-1,:3]
    colors = np.stack((0.*np.ones(3), np.array([0.5,0,0]), 
                        np.array([0,0,1]), *gs), axis=0)


    isort0, isort_split, isort_tsne = isorts[:3]

    dx = 0.01
    ax = plt.subplot(grid[0,0])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+2*dx, pos[1], pos[2], pos[3]])
    transl = mtransforms.ScaledTranslation(-40 / 72, 10 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    line = plot_sorting2d(ax, xi, isort0, label=True)
    ax.text(-0.15, 1.07, "sorting by: ", fontweight="bold", transform=ax.transAxes)
    ax.set_title("   rastermap", fontsize="medium", 
                color=colors[0], loc="center")#, fontweight="bold")
    cax = ax.inset_axes([-0.04, 0.0, 0.03, 0.3])
    cmap = cmap_emb(np.linspace(0.1,1.0,50))[:,np.newaxis,:3]
    cax.imshow(cmap, aspect="auto")
    #cax.axis("off")
    cax.set_xticks([])
    cax.set_yticks([0, 50])
    cax.set_yticklabels(["1st", "last"])

    ax = plt.subplot(grid[0,1])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+dx/2, pos[1], pos[2], pos[3]])
    plot_sorting2d(ax, xi, isort_split)
    ax.set_title("rastermap + splitting", fontsize="medium",
                color=colors[1], loc="center")#, fontweight="bold")

    ax = plt.subplot(grid[0,2])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]-dx, pos[1], pos[2], pos[3]])
    plot_sorting2d(ax, xi, isort_tsne)
    ax.set_title("t-SNE multi-perplexity", fontsize="medium", 
                color=colors[2], loc="center")#, fontweight="bold")

    ax = plt.subplot(grid[0,3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+dx, pos[1]+0.03, pos[2]-dx, pos[3]])
    transl = mtransforms.ScaledTranslation(-40 / 72, 2 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    titles = ["rastermap", "+ splitting", "t-SNE multi-$P$"]
    for perplexity in perplexities[1:]:
        titles.append(f"{perplexity[0]}")
    ax.text(1, 0.35, "t-SNE $P$ =     ", color=colors[3], transform=ax.transAxes, ha="right")
    for i in range(knn_score.shape[1]):
        ax.semilogx(knn, knn_score[:,i]*100, color=colors[i], lw=2+(i<3), zorder=7-i)
        ax.text(1, 0.35-(i-3)*0.1, titles[i], color=colors[i], transform=ax.transAxes, ha="right")
    ax.set_ylim([0, 68])
    ax.set_ylabel("% neighbors preserved")
    ax.set_xlabel("neighborhood size")
    ax.set_xticks([10,100,1000])
    ax.set_xticklabels(["10","100","1000"])
    #plot_raster(ax, X_embedding, 200, 275, vmax=2)

    ax = plt.subplot(grid[0,4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0]+dx, pos[1]+0.01, pos[2]-dx, pos[3]])
    transl = mtransforms.ScaledTranslation(-20 / 72, 5 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    im = ax.imshow(X_embedding[:, 230:320], vmin=0, vmax=2, 
                cmap="gray_r", aspect="auto")
    ax.set_title("sorted activity", fontsize="medium")
    ax.set_xlabel("time")
    ax.set_ylabel("superneurons")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    return il

def panels_v1stimresp(fig, grid, il, yratio, areas, X_embedding, bin_size,
                      isort, xpos, ypos, stim_times, run, ex_stim, 
                      isort2, x, rfs, g0=1):
    subsample = 5
    xpos_sub = xpos[::subsample]
    ypos_sub = ypos[::subsample]

    ax = plt.subplot(grid[g0,0])
    pos = ax.get_position().bounds
    pos_ratio = (ypos.max() - ypos.min()) / (xpos.max() - xpos.min())
    ax.set_position([pos[0]+0.01, pos[1]-0.07, pos[2]*0.9, pos[2]*0.9*yratio*pos_ratio])    
    memb = np.zeros_like(isort)
    memb[isort] = np.arange(0, len(isort))
    memb = memb[::subsample]
    ax.scatter(xpos_sub, -ypos_sub, s=2.5, 
            c=memb, cmap=cmap_emb, alpha=0.15, rasterized=True)
    iarea = [21, 23, 27, 3, 5]
    xw = 10
    x0, y0 = -1100, -1400
    for i in iarea:
        xx, yy = (areas[i][0]-680)/1, (1050-areas[i][1])/1
        if i==5:
            slc = slice(280,400)
            xx = xx[slc]
            yy = yy[slc]
        elif i==3:
            slc = slice(0,200)
            xx = xx[slc]
            yy = yy[slc]
        ax.plot(xx*xw + x0 + xpos.min(), 
                yy*xw + y0 + (-ypos).min(), color="k")
    ax.set_xlim([xpos.min(), xpos.max()])
    ax.set_ylim([(-ypos).min(), (-ypos).max()])
    ax.text(0.2, 0.3, "V1", transform=ax.transAxes, fontsize="large", fontweight="bold")
    ax.text(0.5, 0.8, "RL", transform=ax.transAxes, fontsize="large", fontweight="bold")
    ax.text(0.85, 0.5, "AL", transform=ax.transAxes, fontsize="large", fontweight="bold")
    ax.text(0.8, 0.13, "LM", transform=ax.transAxes, fontsize="large", fontweight="bold")
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])

    axin = ax.inset_axes([0.87, 1.1, 0.1, 0.1])
    add_apml(axin, 0*np.ones(1), 0*np.ones(1), dx=200, dy=200)
    axin.axis("off")

    axin = ax.inset_axes([-0.0, 1.05, 0.7, 0.4])
    transl = mtransforms.ScaledTranslation(-15 / 72, -7 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, axin, transl, fs_title)
    dx = 50
    dy = 20
    nimg, ly, lx = ex_stim.shape
    imgs  = 255 * np.ones((ly + (nimg-1)*dy, lx + (nimg-1)*dx), "uint8")
    for i in range(nimg):
        x0, y0 = i*dx, i*dy
        imgs[y0:y0+ly, x0:x0+lx] = ex_stim[nimg-1-i]
    axin.imshow(imgs, cmap="gray", vmin=0, vmax=255)
    axin.text(1, -0.05, "x5000", fontweight="bold", 
            ha="right", va="top", transform=axin.transAxes)
    axin.axis("off")

    xmin = 5000
    xmax = 5400
    padding_x = 0.005
    
    ax = plt.subplot(grid[g0,1:4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]-0.03, pos[2], pos[3]*0.9])    
    axs = fig.add_axes([pos[0], pos[1]-0.03 + pos[3]*1.07, pos[2], pos[3]*0.05])
    axs.text(0, 1.1, "visual stim.", transform=axs.transAxes)
    st = stim_times[np.logical_and(stim_times < xmax, stim_times >= xmin)] - xmin
    axs.plot([0, xmax-xmin], [0, 0], color="k")
    axs.scatter(st, np.zeros(len(st)), color="k", marker=2, rasterized=True)
    axs.set_xlim([0, (xmax-xmin)*(1+padding_x*2)])
    axs.set_ylim([0, 1])
    axs.axis("off")
    transl = mtransforms.ScaledTranslation(-15 / 72, -2 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, axs, transl, fs_title)
    axr = fig.add_axes([pos[0], pos[1]-0.03 + pos[3]*0.92, pos[2], pos[3]*0.13])
    axr.fill_between(np.arange(0, xmax-xmin), run[xmin : xmax], color=kp_colors[0])
    axr.text(0, 0.55, "running speed", color=kp_colors[0], 
            transform=axr.transAxes)
    axr.set_xlim([0, (xmax-xmin)*(1+padding_x*2)])
    axr.axis("off")
    pos = ax.get_position().bounds
    cax = fig.add_axes([pos[0]+0.6*pos[2], pos[1]-pos[3]*0.05, 
                        pos[2]*0.06, pos[3]*0.03])
    plot_raster(ax, X_embedding, xmin, xmax, cax=cax, cax_label="left", 
                vmax=1.5, padding=0.04, padding_x=padding_x, 
                nper=bin_size, n_neurons=5000,
                fs=3.2, label=True, label_pos="right")
    cax = fig.add_axes([pos[0]-pos[2]*0.03, pos[1], pos[2]*0.015, pos[3]])
    nn = X_embedding.shape[0]
    emb_cols = plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95, nn))[::-1]
    cax.imshow(emb_cols[:,np.newaxis], aspect="auto")
    cax.set_ylim([0, nn*1.025])
    cax.invert_yaxis()
    cax.axis("off")

    ax = plt.subplot(grid[g0+1,1:4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]-0.02, pos[2], pos[3]*0.9])    
    transl = mtransforms.ScaledTranslation(-15 / 72, 5 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    plot_raster(ax, x[:,isort2][:,::10], xmin=0, xmax=500, vmax=1.5, padding=0.04,
                padding_x=padding_x, nper=bin_size, n_neurons=5000, xlabel="stim.",
                label_pos="right", label=True, fs=1, n_sec=20)
    ax.set_title("responses sorted across stimuli", fontsize="medium")

    ax = plt.subplot(grid[g0:g0+4, 4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]-0.07, pos[2], pos[3]+0.05])    
    ny, nx = 30, 8
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(ny,nx, subplot_spec=ax, 
                                                                wspace=0.15, hspace=0.1)
    ax.remove()
    dj = nn // (nx*ny)
    vmax = 1e-4
    rfj = np.linspace(0, nn-1, nx*ny).astype("int")
    for j in range(nx*ny):
        ax = plt.subplot(grid1[j//nx, j%nx])
        rfi = rfj[nx*ny - 1 - j]
        ax.imshow(rfs[rfi], vmin = -vmax, vmax = vmax, cmap = 'RdBu_r', rasterized=True)
        if j==0:
            ax.text(0, 2.5, "superneuron receptive fields", transform=ax.transAxes)
            transl = mtransforms.ScaledTranslation(-15 / 72, 15 / 72, fig.dpi_scale_trans)
            il = plot_label(ltr, il, ax, transl, fs_title)
        ax.axis('off')
        if j<2 or j>nx*ny-3:
            ax.text(0.5, 1.1, f"{nn - rfi}", ha="center", transform=ax.transAxes)
        elif j==2 or j==nx*ny-3:
            ax.text(0.5, 1.5, f"...", ha="center", transform=ax.transAxes)


    return il

def panels_alexnet(fig, grid, il, X_embedding, bin_size, 
                   isort, isort2, ipos, ilayer, iconv, nmax, g0=2):
    
    memb = np.zeros(len(isort))
    memb[isort] = np.arange(len(isort))
    ax = plt.subplot(grid[g0,0])
    transl = mtransforms.ScaledTranslation(-15 / 72, -12 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    ypre = 0
    for j in range(5):
        inds = slice(j*(nmax), (j+1)*(nmax), 2)
        xp, yp = ipos[inds][:,::-1].T
        ax.scatter(5*j + xp - xp.mean() + np.random.randn(len(xp))/4, 
                    -ypre + yp - yp.max() + np.random.randn(len(xp))/4 , 
                    c=memb[inds], cmap=cmap_emb, vmin=0, vmax=len(isort), 
                    s=5-j, alpha=0.25, rasterized=True)
        ax.text(5*j + 0.5 + xp.max() - xp.mean(), -ypre - yp.max() - 0.5, 
                f"conv{j+1}", ha="right", va="top")
        ypre += (yp.max() + 4)
    ax.axis("square")
    ax.axis("off")
    ax.set_title("Alexnet convolutional layer responses", y=0.88)

    ax = plt.subplot(grid[g0,1:4])
    xmin = 2200
    xmax = 2700
    padding_x = 0.005
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]-0.01, pos[2], pos[3]*0.9]) 
    pos = ax.get_position().bounds
    cax = fig.add_axes([pos[0]+0.6*pos[2], pos[1]-pos[3]*0.05, 
                        pos[2]*0.06, pos[3]*0.03])
    plot_raster(ax, X_embedding, xmin, xmax, cax=cax, cax_label="left", 
                vmax=2, padding=0.04, padding_x=padding_x, 
                nper=bin_size, n_neurons=1000, xlabel="stim.",
                fs=1, n_sec=20, label=True, label_pos="right")
    #ax.set_title("Alexnet responses to visual stimuli")
    cax = fig.add_axes([pos[0]-pos[2]*0.03, pos[1], pos[2]*0.015, pos[3]])
    nn = X_embedding.shape[0]
    emb_cols = plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95, nn))[::-1]
    cax.imshow(emb_cols[:,np.newaxis], aspect="auto")
    cax.set_ylim([0, nn*1.025])
    cax.invert_yaxis()
    cax.axis("off")

    ax = plt.subplot(grid[g0+1,1:4])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]-0.0, pos[2], pos[3]*0.9])    
    transl = mtransforms.ScaledTranslation(-15 / 72, 5 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)
    plot_raster(ax, X_embedding[:,isort2][:,::10], xmin=0, xmax=500, vmax=1.5, padding=0.04,
                padding_x=padding_x, nper=bin_size, n_neurons=5000, xlabel="stim.",
                label_pos="right", label=True, fs=1, n_sec=20)
    ax.set_title("responses sorted across stimuli", fontsize="medium")


    return il

    
def fig_all(root, save_figure=True):
    
    fig = plt.figure(figsize=(14,8))
    yratio = 14 / 8
    grid = plt.GridSpec(3,5, figure=fig, left=0.01, right=0.99, top=0.94, bottom=0.055, 
                    wspace = 0.15, hspace = 0.25)
    il = 0

    try:
        d = np.load(os.path.join(root, "simulations", "sim2D_results.npz"))
        il = panels_sim2d(fig, grid, il, **d)
    except:
        print("simulation data not available")

    try:
        areas = loadmat(os.path.join(root, "figures", "ctxOutlines.mat"), 
                        squeeze_me=True)["coords"]
        d = np.load(os.path.join(root, "results", "v1stimresp_proc.npz"))
        il = panels_v1stimresp(fig, grid, il, yratio, areas, **d)
    except:
        print("visual data not available")

    try:
        d = np.load(os.path.join(root, "results", "alexnet_proc.npz"))
        il = panels_alexnet(fig, grid, il, **d)  
    except:
        print("alexnet data not available")

    if save_figure:
        fig.savefig(os.path.join(root, "figures", "fig6.pdf"), dpi=200)