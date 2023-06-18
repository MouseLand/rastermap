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
                label_pos="left", axis_off=False):
    xr = xmax - xmin
    nn = X.shape[0]
    im = ax.imshow(X[:, xmin:xmax], vmin=-vmax if symmetric else 0, vmax=vmax, 
              cmap="RdBu_r" if symmetric else "gray_r", aspect="auto")
    ax.axis("off")
    if label_pos=="left":
        ax.plot(-0.005*xr * np.ones(2), nn - np.array([0, n_neurons/nper]), color="k")
        ax.plot(np.array([0, fs*n_sec]), nn*(1+padding/2) + np.zeros(2), color="k")
        ax.set_xlim([-0.008*xr, xr])
    else:
        ax.plot(1.005*xr * np.ones(2), nn - np.array([0, n_neurons/nper]), color="k")
        ax.plot(np.array([xr-fs*n_sec, xr]), nn*(1+padding/2) + np.zeros(2), color="k")
        ax.set_xlim([0, 1.008*xr])
    ax.set_ylim([0, nn*(1+padding)])
    ax.invert_yaxis()
    if cax is not None:
        plt.colorbar(im, cax, orientation="horizontal")
        cax.set_xlabel("z-scored\n ")
    if label:
        if label_pos=="left":
            ht=ax.text(-0.008*xr, X.shape[0], "500 neurons", ha="right")
            ht.set_rotation(90)
            ax.text(0, nn*(1+padding), "10 sec.", va="top")
        else:
            ht=ax.text(1.015*xr, X.shape[0], "500 neurons", ha="left")
            ht.set_rotation(90)
            ax.text(xr-fs*n_sec, nn*(1+padding), "10 sec.", va="top")

