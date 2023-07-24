"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import numpy as np
import matplotlib.pyplot as plt 

from fig_utils import *


def fig5(root, save_figure=True):
    env_ids = ["PongNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4", 
               "EnduroNoFrameskip-v4", "SeaquestNoFrameskip-v4"]
    fig = plt.figure(figsize=(14,7))
    grid = plt.GridSpec(2,6, figure=fig, left=0.04, right=0.98, top=0.96, bottom=0.07, 
                        wspace = 0.15, hspace = 0.25)
    transl = mtransforms.ScaledTranslation(-13 / 72, 20 / 72, fig.dpi_scale_trans)
    il = 0
    layer_cols = cmap_emb(np.array([0.55, 0.65, 0.75, 0.9, 0]))
    layer_names = ["conv1", "conv2", "conv3", "linear", "valuenet"]
    for igame, env_id in enumerate(env_ids):
        print(env_id)
        i0, j0 = igame//2, 3*(igame%2)

        d = np.load(os.path.join(root, "simulations/", f"qrdqn_{env_id}_results.npz"))
        X_embedding = d["X_embedding"]
        nn, nt = X_embedding.shape
        emb_layer = d["emb_layer"]
        ex_frames = d["ex_frames"]
        iframes = d["iframes"]

        ax = plt.subplot(grid[i0, j0+1:j0+3])
        pos = ax.get_position().bounds
        ax.imshow(X_embedding, aspect='auto', 
                 vmax=2.5, vmin=-0., 
                 cmap='gray_r')
        
        for k in range(4):
            ik = iframes[k]
            ax.plot(ik*np.ones(2), [0, nn], color="b", ls="--")
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])
        ax.set_ylim([0, nn])
        if env_id=="EnduroNoFrameskip-v4":
            ax.set_xlim([780, nt])
        ax.invert_yaxis()
        ax.set_xlabel("timepoint in episode")
        if igame==0:
            ax.text(0.28, 1.02, "layers in DQN: ", color="k", 
                    transform=ax.transAxes, ha="right")
            for l, lcol in enumerate(layer_cols):
                ax.text(0.3+l*0.13, 1.02, layer_names[l], color=lcol, transform=ax.transAxes)
                if l<4:
                    ax.text(0.3+(l+1)*0.13-0.02, 1.02, ",", color="k", transform=ax.transAxes)

        cax = fig.add_axes([pos[0]+pos[2]*1.015, pos[1], pos[2]*0.015, pos[3]])
        cax.imshow(layer_cols[emb_layer][:,np.newaxis], aspect="auto")
        cax.axis("off")

        ax = plt.subplot(grid[i0,j0])
        pos = ax.get_position().bounds
        ax.set_position([pos[0]+0.08*pos[2], pos[1]-0.08*pos[3], pos[2], pos[3]])
        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec=ax, 
                                                            wspace=0.05, hspace=0.15)
        ax.remove()
        for k in range(4):
            ax = plt.subplot(grid1[k//2, k%2])
            ax.imshow(ex_frames[k])
            ax.set_title(f"frame {iframes[k]}", fontsize="medium", color="b")
            ax.axis("off")
            if k==0:
                ax.text(0, 1.23, env_id[:-14], fontsize="large", transform=ax.transAxes)
                il = plot_label(ltr, il, ax, transl, fs_title)

    if save_figure:
        fig.savefig(os.path.join(root, "figures", "fig5.pdf"), dpi=200)