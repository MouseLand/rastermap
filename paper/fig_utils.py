"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import string
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

cmap_emb = ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95), 100))


kp_colors = np.array([[0.55,0.55,0.55],
                      [0.,0.,1],
                      [0.8,0,0],
                      [1.,0.4,0.2],
                      [0,0.6,0.4],
                      [0.2,1,0.5],
                      ])

default_font = 12
rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titlelocation"] = "left"
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = default_font

ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"


def add_apml(ax, xpos, ypos, dx=300, dy=300, tp=30):
    x0, x1, y0, y1 = (
        xpos.min() - dx / 2,
        xpos.min() + dx / 2,
        ypos.max(),
        ypos.max() + dy,
    )
    ax.plot(np.ones(2) * (y0 + dy / 2), [x0, x1], color="k")
    ax.plot([y0, y1], np.ones(2) * (x0 + dx / 2), color="k")
    ax.text(y0 + dy / 2, x0 - tp, "P", ha="center", va="top", fontsize="small")
    ax.text(y0 + dy / 2, x0 + dx + tp, "A", ha="center", va="bottom", fontsize="small")
    ax.text(y0 - tp, x0 + dx / 2, "M", ha="right", va="center", fontsize="small")
    ax.text(y0 + dy + tp, x0 + dx / 2, "L", ha="left", va="center", fontsize="small")
    print(x0, y0)

def plot_label(ltr, il, ax, trans, fs_title=20):
    ax.text(
        0.0,
        1.0,
        ltr[il],
        transform=ax.transAxes + trans,
        va="bottom",
        fontsize=fs_title,
        fontweight="bold",
    )
    il += 1
    return il


def plot_raster(ax, X, xmin, xmax, vmax=1.5, symmetric=False, cax=None, nper=30, 
                label=False, n_neurons=500, n_sec=10, fs=20, padding=0.025,
                padding_x = 0.005, xlabel="sec.",
                label_pos="left", axis_off=False, 
                cax_label="x", cax_orientation="horizontal"):
    xr = xmax - xmin
    nn = X.shape[0]
    if n_neurons is None:
        xmin0, xmax0 = 0, X.shape[1]
    else:
        xmin0, xmax0 = xmin, xmax
        xmin, xmax = 0, xmax - xmin
    im = ax.imshow(X[:, xmin0:xmax0], vmin=-vmax if symmetric else 0, vmax=vmax, 
              cmap="RdBu_r" if symmetric else "gray_r", aspect="auto")
    ax.axis("off")
    if label_pos=="left":
        if n_neurons is not None:
            ax.plot(-padding_x*xr * np.ones(2), nn - np.array([0, n_neurons/nper]), color="k")
        ax.plot(xmin + np.array([0, fs*n_sec]), nn*(1+padding/2) + np.zeros(2), color="k")
    else:
        if n_neurons is not None:
            ax.plot((1+padding_x)*xr * np.ones(2), nn - np.array([0, n_neurons/nper]), color="k")
        ax.plot(xmin + np.array([xr-fs*n_sec, xr]), nn*(1+padding/2) + np.zeros(2), color="k")
    ax.set_ylim([0, nn*(1+padding)])
    ax.invert_yaxis()
    if cax is not None:
        plt.colorbar(im, cax, orientation=cax_orientation)
        if cax_label=="x":
            cax.set_xlabel("z-scored\n ")
        else:
            cax.text(-0.2,0,"z-scored", transform=cax.transAxes, 
                    ha="right", 
                    rotation=90 if cax_orientation=="vertical" else 0)
    if n_neurons is None:
        ax.set_xlim([xmin, xmax])
    else:
        if label_pos=="left":
            ax.set_xlim([-2*padding_x*xr, xr])
        else:
            ax.set_xlim([0*xr, xr*(1+padding_x*2)])
    if label:
        if label_pos=="left":
            if n_neurons is not None:
                ht=ax.text(-2*padding_x*xr, X.shape[0], f"{n_neurons} neurons", ha="right")
                ht.set_rotation(90)
            ax.text(xmin, nn*(1+padding), f"{n_sec} {xlabel}", va="top")
        else:
            if n_neurons is not None:
                ht=ax.text((1+2*padding_x)*xr, X.shape[0], f"{n_neurons} neurons", ha="left")
                ht.set_rotation(90)
            ax.text(xr, nn*(1+padding), f"{n_sec} {xlabel}", 
                    va="top", ha="right")
