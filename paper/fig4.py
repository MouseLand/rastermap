"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
from fig_utils import *
from matplotlib import patches

def brain_plot(ax, x, y, cweights, cmap, theta=np.pi*0.77, subsample=5, 
                vmin=None, vmax=None, brain_axes=True):
    pos = ax.get_position().bounds
    xvox = x*np.cos(theta) - y*np.sin(theta)
    yvox = x*np.sin(theta) + y*np.cos(theta)
    im = ax.scatter(-xvox[::subsample], yvox[::subsample], 
                cmap=cmap, c=cweights[::subsample], rasterized=True,
                s=1, alpha=1, marker=".", vmin=vmin, vmax=vmax)
    if brain_axes:
        il = add_apml(ax, 210*np.ones(1), 45*np.ones(1), dx=80, dy=80, tp=10)
    ax.axis("off")
    ax.set_xlim([0, (-xvox).max()*1.1])
    ax.set_ylim([-3*yvox.max(), 2*yvox.max()])
    if isinstance(cmap, str):
        cax = ax.figure.add_axes([pos[0]+pos[2]*0.3, pos[1]+pos[3]*.35, 
                                    pos[2]*0.5, pos[3]*0.04])
        plt.colorbar(im, cax=cax, orientation="horizontal")
        if cmap=="viridis":
            cax.set_xlabel("var. exp.")
        else:
            cax.set_xticks([-0.2,0,0.2])
            cax.set_xticklabels(["-0.2  ", "0", "  0.2"])
            cax.set_xlabel("diff. in \nvar. exp.")

    #ax.axis("square")

def panels_hippocampus(fig, grid, il, spks, pyr_cells, speed, loc2d, tcurves, isort,
                        cc_nodes, isort2, xmin=1000, xmax=2000, bin_sec=0.2):
    ax = plt.subplot(grid[0,:3])
    pos = ax.get_position().bounds
    ax.set_position([pos[0], pos[1]+pos[3]*0.1, pos[2], pos[3]*0.9])
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(8,5, subplot_spec=ax, 
                                                            wspace=0.3, hspace=0.1)
    ax.remove()

    nn = spks.shape[0]
    xr = xmax - xmin
    padding = 0.025

    ax = plt.subplot(grid1[0,:-1])
    sp = loc2d[xmin:xmax].copy()
    sp -= sp.min()
    spm = sp.max()
    sp /= spm
    sp[:,1] += sp[:,0].mean() - sp[:,1].mean()
    d = -0.005*xr
    for j in range(2):
        ax.plot(sp[:,j], color=kp_colors[j+2])
        ax.text(0.06+j*.01, 0.9-j*0.34,"x-position" if j==0 else "y-position", 
                transform=ax.transAxes, color=kp_colors[j+2])
    ax.plot(d*np.ones(2), [0,1/(spm)], color="k")
    ht=ax.text(-0.008*xr, 0.15, "1 m", ha="right")
    ht.set_rotation(90)
    ax.set_xlim([-0.008*xr, xr])
    ax.axis("off")

    ### title
    ax.text(0, 1.45, "rat hippocampus ephys recording, 1.6m linear track", 
                    transform=ax.transAxes, fontsize="large")
    transl = mtransforms.ScaledTranslation(-15 / 72, 14/ 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, transl, fs_title)

    ax = plt.subplot(grid1[1,:-1])
    sp = speed[xmin:xmax].copy()
    spm = sp.max()
    sp /= spm
    ax.fill_between(np.arange(0, xr), sp, color=kp_colors[0])
    d = -0.005*xr
    ax.plot(d*np.ones(2), [0, 0.5/(spm/bin_sec)], color="k")
    ht=ax.text(-0.008*xr, -0.15, "0.5 m/s", ha="right")
    ht.set_rotation(90)
    ax.set_xlim([-0.008*xr, xr])
    ax.text(0.08, 0.75, "running speed", transform=ax.transAxes, color=kp_colors[0])
    ax.axis("off")

    ax = plt.subplot(grid1[2:,:-1])
    pos = ax.get_position().bounds
    xw = pos[2]*0.05
    cax = fig.add_axes([pos[0]+pos[2]-xw, pos[1]-pos[3]*0.015, xw, pos[3]*0.025])
    plot_raster(ax, spks[isort], xmin, xmax, nper=1, n_neurons=20, padding=padding,
                n_sec=10, fs=1/bin_sec, label=True, cax=cax)

    cax = fig.add_axes([pos[0]+pos[2]*1.01, pos[1], pos[2]*0.01, pos[3]])
    cols = np.zeros((nn, 3))
    cols[pyr_cells] = np.array([0,1,0])
    cols[~pyr_cells] = np.array([0,0,1])
    cax.imshow(cols[isort,np.newaxis], aspect="auto")
    cax.text(1.1, 0.93, "FS", transform=cax.transAxes, color=[0,0,1])
    cax.text(1.1, 0.75, "RS", transform=cax.transAxes, color=[0,1,0])
    cax.set_ylim([0, (1+padding)*nn])
    cax.invert_yaxis()
    cax.axis("off")

    ax = plt.subplot(grid1[2:, -1])
    n_pos = tcurves.shape[1]//2
    x = np.arange(0, n_pos)
    dy = 2
    xpad = n_pos/10
    for t in range(len(tcurves)):
        ax.plot(x, tcurves[isort[t], :n_pos]*dy + (1+padding)*nn - t, 
                color="k", lw=0.5)
        ax.plot(x+n_pos+xpad, tcurves[isort[t], n_pos:]*dy + (1+padding)*nn - t, 
                color="k", lw=0.5)
    for j in range(2):
        xstr = "position\n(left run)" if j==0 else "position\n(right run)"
        ax.text(n_pos/2 + j*(n_pos+xpad), -18, xstr, ha="center")
        ax.text(j*(n_pos+xpad), -3, "0")
        ax.text(n_pos + j*(n_pos+xpad), -3, "1.6", ha="right")

    ax.set_title("single-neuron\ntuning curves", loc="center")
    ax.set_ylim([0,nn*(1+padding)])
    ax.set_xlim([0.05, 2*n_pos-0.05])
    ax.axis("off")

    return il

def panels_widefield(fig, grid, il, stim_times_0, stim_times_1, 
                    stim_times_2, stim_times_3, 
                    stim_labels, reward_times, sn, sn_pred, sn_pred_beh, 
                     bin_size, itest, ypos, xpos, isort, 
                    xmin=820, xmax=1730):
    stim_times = [stim_times_0, stim_times_1, stim_times_2, stim_times_3]

    titles = ["(i) widefield imaging during a decision-making task",
                "(ii) prediction of activity from task and behavior variables",
                "(iii) prediction of activity from behavior variables only",
                "(iv) difference between (ii) and (iii)"]

    ax = plt.subplot(grid[:,3:])
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(4,5, subplot_spec=ax, 
                                                            wspace=0.25, hspace=0.2)
    ax.remove()

    nn = sn.shape[0]
    padding = 0.03

    ven = 1 - (((sn[:,itest] - sn_pred)**2).mean(axis=1) / 
                (sn**2).mean(axis=1))

    ven_beh = 1 - (((sn[:,itest] - sn_pred_beh)**2).mean(axis=1) / 
                    (sn**2).mean(axis=1))

    memb = np.zeros_like(isort)
    memb[isort] = np.arange(0, len(isort))
    subsample = 10

    ns = (len(ypos)//bin_size) * bin_size
    ven_t = np.repeat(ven, bin_size).T.flatten()
    ven_beh_t = np.repeat(ven_beh, bin_size).T.flatten()

    ypos_sort = ypos[isort][:ns]
    xpos_sort = xpos[isort][:ns]
    transl = mtransforms.ScaledTranslation(-15 / 72, 10/ 72, fig.dpi_scale_trans)
    for j in range(4):
        ax = plt.subplot(grid1[j,0])
        if j==0:
            cweights=memb
            cmap = cmap_emb
            vmin, vmax = 0, len(isort)
        elif j<3:
            vmin, vmax = 0, 0.55
            cmap = "viridis"
            cweights = ven_t if j==1 else ven_beh_t
        else:
            vmin, vmax = -0.2, 0.2
            cmap = "RdBu_r"
            cweights = ven_t - ven_beh_t 
        brain_plot(ax, ypos_sort if j>0 else ypos, xpos_sort if j>0 else xpos, 
                    cweights=cweights, cmap=cmap, subsample=subsample, 
                    vmin=vmin, vmax=vmax, brain_axes=(j==0))
        ax.set_title(titles[j])
        if j==0:
            il = plot_label(ltr, il, ax, transl, fs_title)


    for j in range(4):
        if j==0:
            X = sn[:, itest]
        elif j==1:
            X = sn_pred
            ve = ven
        elif j==2:
            X = sn_pred_beh 
            ve = ven_beh
        else:
            X = sn_pred - sn_pred_beh
            ve = ven - ven_beh
        vmax=1.1
        ax = plt.subplot(grid1[j,1:])
        poss = ax.get_position().bounds
        ax.set_position([poss[0], poss[1], poss[2]*0.9, poss[3]])
        poss = ax.get_position().bounds
        cax = fig.add_axes([poss[0]+poss[2]*1.06, poss[1]+0.05*poss[3], 
                            poss[2]*0.02, 0.2*poss[3]])
        plot_raster(ax, X, xmin, xmax, n_neurons=None, fs=30, n_sec=5,
                    padding=padding, cax=cax, cax_orientation="vertical",
                    cax_label="left",   vmax=vmax,
                    symmetric=(j==3), label=(j==0))

        if j < 3:
            nn = sn.shape[0]
            reward_color = [0,0.5,0]
            fcolor = np.array([[0,0.5,1], 
                            [0,1,1], 
                            [1,0,0], 
                            [1,0.5,0],
                            [0.8,0.5,0.7]])
            for k in range(len(stim_times)):
                starts = stim_times[k] - itest[0]
                if j==0:
                    ax.text(1.02,0.95-k*0.1, stim_labels[k], 
                            color=fcolor[k], transform=ax.transAxes)
                for n in range(len(starts)-1):
                    start = starts[n]+1
                    width = 1.6*30
                    # add stimulus patch
                    ax.add_patch(
                            patches.Rectangle(xy=(start, 0), width=width,
                                    height=nn, facecolor=fcolor[k], 
                                    edgecolor=None, alpha=0.2))
                    
            for reward_time in reward_times:
                ax.plot([reward_time, reward_time], [0, nn],
                            color=reward_color, lw=2)
            if j==0:
                ax.text(1.02,0.95-(k+1)*0.1, "reward", 
                            color=reward_color, transform=ax.transAxes)

        cax = fig.add_axes([poss[0]-poss[2]*0.04, poss[1], poss[2]*0.02, poss[3]])
        cols = cmap_emb(np.linspace(0, 1, nn))
        if j==0:
            cax.imshow(cols[:,np.newaxis], aspect="auto")
        else:
            if j<3:
                vmin, vmax = 0, 0.55
                cmap = "viridis"
            else:
                vmin, vmax = -0.2, 0.2
                cmap = "RdBu_r"
            cax.imshow(ve[:,np.newaxis], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            
        cax.set_ylim([0, (1+padding)*nn])
        cax.invert_yaxis()
        cax.axis("off")
        cax.add_patch(patches.Rectangle(xy=(-0.5, 0), width=0.99, height=nn*1, 
                                fill=False, edgecolor="k", lw=1))
    return il


def panels_fish(fig, grid1, il, sn, swimming, eyepos, 
                stims, xyz, isort, isort2, cc_nodes):
    nn = sn.shape[0]
    nx = 3
    ny = 6
    nxy = nx * ny

    xi = np.hstack((np.arange(5700, 6500), np.arange(7300,7860)))
    stims = stims[xi]
    #xmin=5600, xmax=7900

    ax = plt.subplot(grid1[0,:-1])
    pos = ax.get_position().bounds 
    ax.remove()

    yh = 0.09*pos[3]
    yh2 = 0.05*pos[3]
    ypad = 0.01*pos[3]
    xp = 0.025*pos[2]
    ax_neural = fig.add_axes([pos[0], pos[1]+2*yh+2*ypad, 
                        pos[2]-xp, pos[3]-(2*yh+yh2+3*ypad)])
    pos_neural = ax_neural.get_position().bounds

    ### eyepos
    ax = fig.add_axes([pos_neural[0], pos[1], pos_neural[2], yh])
    eye_color = [[0,0.95,0], [0,0.5,0.]]
    for i in range(2):
        ax.plot(eyepos[xi,i], color=eye_color[i])
        ax.text(320+130*i, eyepos[xi].max()*(0.8), 
                "left" if i==0 else "right", color=eye_color[i])
    ax.text(0, eyepos[xi].max()*0.8, "eye pos.", 
                color="k")
    ax.plot([0, 60], (-1+eyepos[xi].min())*np.ones(2), 
                color="k")
    ax.text(0, -5, "30 sec.")
    ax.set_xlim([0, len(xi)])
    ax.axis("off")
    
    ### swimming
    swim_color = np.array([[0,1,1,1],
                           [0,0,1,1],
                           [1.0,0.0,0,1],
                           [0.8,1.0,0,1]])

    ax = fig.add_axes([pos_neural[0], pos[1] + yh + ypad, 
                        pos_neural[2], yh])
    for i in range(2):
        ax.plot(swimming[xi,i], color=swim_color[i])
        ax.text(0+320+130*i, swimming[xi].max()*(0.6), 
                "left" if i==0 else "right", color=swim_color[i])
    ax.text(0, swimming[xi].max()*0.6,"swimming", 
                color="k")
    ax.set_ylim([swimming[xi].min(), swimming[xi].max()])
    ax.set_xlim([0, len(xi)])
    ax.axis("off")

    ### neuron activity
    ax_neural.imshow(sn[:, xi], vmin=0, vmax=1, 
                  cmap="gray_r", aspect="auto")
    ax_neural.axis("off")
    ax = fig.add_axes([-pos_neural[2]*0.01+pos_neural[0], pos_neural[1], 
                        0.01*pos_neural[2], pos_neural[3]])
    ax.plot(np.ones(2), [0, 40], 
                color="k")
    ax.text(-2.5, 0, "2000 neurons", transform=ax.transAxes, rotation=90, ha="left")
    ax.axis("off")
    ax.set_ylim([0, nn])

    ### stims
    ax_stim = fig.add_axes([pos_neural[0],pos_neural[1]+pos_neural[3]+ypad,
                            pos_neural[2], yh2])

    fcolor = plt.get_cmap("hsv")(np.linspace(0,1,stims.max()+1))[::-1]
    fcolor[0] = np.array([0., 0.5, 1.0, 1.0])
    fcolor[1] = np.array([1.0, 0.0, 1.0, 1.0])
    fcolor[2] = np.array([1., 1., 1., 1.])
    fcolor[8] = swim_color[2]
    fcolor[9] = swim_color[3]
    fcolor[10] = swim_color[1]
    fcolor[11] = swim_color[0]
    
    starts = np.nonzero(np.diff(stims))
    starts = np.append(np.array([0]), starts)
    starts = np.append(starts, np.array([len(stims)-1]))
    for n in range(len(starts)-1):
        start = starts[n]+1
        stype = stims[start]
        if stype!=3:
            width = starts[n+1] - start
            width += min(0, start)
            start = max(0, start)
            width = min(width, len(xi) - start)
            ax_neural.add_patch(
                    patches.Rectangle(xy=(start, 0), width=width,
                                height=nn, facecolor=fcolor[stype], 
                                edgecolor=None, alpha=0.15*(stype!=2)))

            if stype==11 or stype==10:
                if stype==11:
                    ax_stim.arrow(start+0.2*width, 0, width*0.75, 0, width=0.04, 
                                    head_length=10, length_includes_head=True,
                                    facecolor=fcolor[stype], edgecolor='none')
                else:
                    ax_stim.arrow(start+width*0.8, 0, -width*0.75, 0, width=0.04, 
                                    head_length=10, length_includes_head=True,
                                    facecolor=fcolor[stype], edgecolor='none')
                pim = np.ones((12,12))
                ax_stim.imshow(pim, extent=(start, start+width+2, -0.1, 0.1), 
                            aspect="auto", cmap="gray", vmin=0, vmax=1)
            elif stype==9 or stype==8:
                if stype==9:
                    ax_stim.arrow(start+width*0.5, -0.06, 0, 0.13*1.2, width=8,
                                    head_length=0.08, length_includes_head=True,
                                    facecolor=fcolor[stype], edgecolor='none')      
                else:
                    ax_stim.arrow(start+width*0.5, 0.08, 0, -0.13*1.2, width=8,
                                head_length=0.08, length_includes_head=True,
                                facecolor=fcolor[stype], edgecolor='none')
                pim = np.ones((12,width))
                ax_stim.imshow(pim, extent=(start, start+width+2, -0.1, 0.1), 
                            aspect="auto", cmap="gray", vmin=0, vmax=1)
            elif stype==1:
                pim = np.ones((12,width))
                ax_stim.text(start+width/2, -0.07, "L", fontsize="small",
                                ha="center", va="bottom", color=fcolor[stype])
                ax_stim.imshow(pim, extent=(start, start+width+1, -0.1, 0.1), 
                            aspect="auto", cmap="gray", vmin=0, vmax=1)
            elif stype==0:
                pim = np.ones((12,width))
                ax_stim.text(start+width/2, -0.07, "R", fontsize="small",
                                ha="center", va="bottom", color=fcolor[stype])
                ax_stim.imshow(pim, extent=(start, start+width+1, -0.1, 0.1), 
                            aspect="auto", cmap="gray", vmin=0, vmax=1)
            elif stype==2:
                pim = np.ones((12,width))
                ax_stim.imshow(pim, extent=(start, start+width+2, -0.1, 0.1), 
                            aspect="auto", cmap="gray", vmin=0, vmax=1)

    ax_stim.set_xlim([0, len(xi)])
    ax_stim.axis("off")

    ax_stim.text(0., 1.1, "phototactic stimuli", 
        transform=ax_stim.transAxes, ha="left")
    ax_stim.text(0.55, 1.1, "optomotor response stimuli", 
        transform=ax_stim.transAxes, ha="center")

    ### title
    ax_stim.text(0, 2, "zebrafish brainwide recording", 
                    transform=ax_stim.transAxes, fontsize="large")
    transl = mtransforms.ScaledTranslation(-15 / 72, 14/ 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax_stim, transl, fs_title)

    
    colors = cmap_emb(np.linspace(0, 1, nxy))

    ### colorbar
    ax = fig.add_axes([pos_neural[2]*1.01+pos_neural[0], pos_neural[1], 
                        0.015*pos_neural[2], pos_neural[3]])
    ax.imshow(colors[:,np.newaxis])
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.set_yticks(np.arange(0,nx*ny+1), 
            labels=[str(i) for i in np.arange(1, nx*ny+2)])
    ax.set_xticks([])
    ax.axis("tight")

    ### neuron locations
    ax = plt.subplot(grid1[0,-1])
    grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(ny, nx, subplot_spec=ax, 
                                                        wspace=0.25, hspace=0.1)
    ax.remove()
    n_neurons = len(isort)
    snp = np.linspace(0, n_neurons, nx*ny+1).astype(int)
    for j in range(nx):
        for k in range(ny):
            ax = plt.subplot(grid2[k,j])
            # plot all neurons
            subsample = 50
            ax.scatter(xyz[:,1][::subsample], xyz[:,0][::subsample], s=2, alpha=1, 
                        color=0.9*np.ones(3), rasterized=True)
            snmin = snp[j + k*nx]
            snmax = snp[j+1 + k*nx]
            ix = isort[snmin : snmax]
            subsample = 3
            ax.scatter(xyz[ix,1][::subsample], xyz[ix,0][::subsample],
                        s=0.5, alpha=0.3, color=colors[j+k*nx])
            ax.axis("off")
            ax.text(0.1,0,str(j+k*nx+1), transform=ax.transAxes, ha="right")
            if j==0 and k==0:
                axin = ax.inset_axes([-0.25, 0.75, 0.2, 0.2])
                add_apml(axin, xyz[:,0], xyz[:,1])
                axin.axis("off")
                axin.axis("tight")
    return il

def fig4(root, save_figure=True):
    
    fig = plt.figure(figsize=(14,10))
    yratio = 14 / 10
    grid = plt.GridSpec(2,5, figure=fig, left=0.02, right=0.98, top=0.96, bottom=0.02, 
                    wspace = 0.2, hspace = 0.08)
    il = 0

    try:
        d = np.load(os.path.join(root, "results", "hippocampus_proc.npz"))
        il = panels_hippocampus(fig, grid, il, **d)    
    except:
        print("hippocampus data not processed")

    ax = plt.subplot(grid[1,:3])
    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec=ax, 
                                                            wspace=0.3, hspace=0.15)
    ax.remove()
    try:
        d = np.load(os.path.join(root, "results", "fish_proc.npz"))
        il = panels_fish(fig, grid1, il, **d) 
    except:
        print("fish data not processed")

    try:
        d = np.load(os.path.join(root, "results", "widefield_proc.npz"))
        il = panels_widefield(fig, grid, il, **d)
    except:
        print("widefield data not processed")

    if save_figure:
        fig.savefig(os.path.join(root, "figures", "fig4.pdf"), dpi=200)

def suppfig_timesort(root, save_figure=True):

    fig = plt.figure(figsize=(14,8))
    yratio = 14 / 8
    grid = plt.GridSpec(2,2, figure=fig, left=0.05, right=0.98, top=0.95, bottom=0.06, 
                    wspace = 0.2, hspace = 0.25)
    il = 0

    titles = ["spontaneous activity", "virtual reality task", "rat hippocampus, linear track",
                "fish wholebrain, visual stimuli"]
    tstr = ["spont", "corridor", "hippocampus", "fish"]
    transl = mtransforms.ScaledTranslation(-20 / 72, 10/ 72, fig.dpi_scale_trans)
        
    for j in range(4):
        d = np.load(os.path.join(root, "results", f"{tstr[j]}_proc.npz"))
        ax = plt.subplot(grid[j//2, j%2])
        il = plot_label(ltr, il, ax, transl, fs_title)
        if j!=2:
            isort2 = d["isort2"]
            sn = d["sn"]
            sp = sn[:,isort2]
        else:
            isort = d["isort"]
            isort2 = d["isort2"]
            spks = d["spks"]
            sp = spks[isort][:,isort2]

        ax.imshow(sp, vmin=0, vmax=1.5, aspect="auto", cmap="gray_r")
        ax.set_xlabel("time sorted")
        ax.set_ylabel("superneurons" if j!=2 else "neurons")
        ax.set_title(titles[j])

    if save_figure:
        fig.savefig(os.path.join(root, "figures", "suppfig_timesort.pdf"), dpi=200)