import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import numpy as np
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

def plot_corridor(S_plot, corridor_starts, corridor_widths, img_corridor, corr_tuning, 
                  RewardInd, SoundInd, LickInd, running, xmin, xmax,
                    ax_events, ax_running, ax_neural, ax_colorbar, ax_img, ax_corr1, ax_corr2):
    axs = [ax_events, ax_running, ax_neural, ax_colorbar, ax_img, ax_corr1, ax_corr2]

    
    axs[2].imshow(zscore(S_plot[:,xmin:xmax], axis=1), vmin=0, vmax=2, cmap='gray_r', aspect='auto')
    axs[2].axis('off')
    
    fcolor = [[0,1,0], [0,0,0.8]]
    for nn in range(len(corridor_starts)):
        if corridor_starts[nn,0]+corridor_widths[nn] > xmin and corridor_starts[nn,0] < xmax:
            icorr = int(corridor_starts[nn,1])
            start = corridor_starts[nn,0]
            width = corridor_widths[nn]
            width += min(0, start-xmin)
            start = max(0, start - xmin)
            width = min(width, xmax - xmin - start)
            axs[2].add_patch(
                Rectangle(xy=(start, 0), width=width,
                          height=S_plot.shape[0], facecolor=fcolor[icorr], edgecolor=None, alpha=0.05))
        
    # add colorbar
    from matplotlib import cm
    plt.colorbar(cm.ScalarMappable(norm=None, cmap='jet_r'), 
                 cax=axs[3])
    axs[3].axis('off')

    # add image of corridor
    upsample = img_corridor.shape[1] // S_plot.shape[1]
    axs[4].imshow(img_corridor[:, xmin*upsample : xmax*upsample], cmap='gray_r', aspect='auto', vmin=25, vmax=255-25)
    axs[4].axis('off')

    for nn in range(len(RewardInd)):
        if RewardInd[nn] > xmin and RewardInd[nn] < xmax:
            start = int(RewardInd[nn] - xmin)
            width = 0
            axs[2].add_patch(
                Rectangle(xy=(start, 0), width=width,
                          height=S_plot.shape[0], facecolor=None, edgecolor='g', alpha=1))
    
    h0=axs[0].scatter(SoundInd,1*np.ones([len(SoundInd),]), 
                      color=[1.,0.6,0], marker='s', s=30)
    h2=axs[0].scatter(LickInd,-1*np.ones([len(LickInd),]), 
                      color=[1.0,0.3,0.3], marker='.', s=30)
    h1=axs[0].scatter(RewardInd,0*np.ones([len(RewardInd),]), 
                      color='g', marker='^', s=30)
    axs[0].axis('off')
    axs[0].set_xlim([xmin, xmax])
    axs[0].set_ylim([-1.35, 1.35])

    axs[0].legend([h0,h1,h2], ['Tone',
                                     'Reward', 
                                     'Licks'
                                     ], loc=(1.01,-3.), fontsize=8)

    axs[1].bar(np.arange(xmin, xmax), running[xmin:xmax], color=[0.5, 0.5, 0.5])
    axs[1].axis('off')
    ymax = running[xmin:xmax].max()
    axs[1].text(xmin, ymax*0.7,'running speed', color=[0.5, 0.5, 0.5])
    axs[1].plot([xmin, xmin+3*10], -0.12*ymax*np.ones(2), color='k')
    axs[1].text(xmin+3*5, -0.2*ymax, '10 sec', ha='center', va='top')
    axs[1].set_xlim([xmin, xmax])
    #axs[1].plot(-0.1+subRot, color=[0.8, 0.8, 0.4])

    nov = 100
    _, n_sn, npts = corr_tuning.shape
    for icorr in range(2):
        ctmax = corr_tuning[icorr].max()
        ctmin = corr_tuning[icorr].min()
        for i in np.arange(0, n_sn, 25, int):
            ct = corr_tuning[icorr, i].copy()
            ct -= ctmin
            ct /= ctmax
            axs[5+icorr].plot(np.arange(0, npts), (n_sn-i-24)+ct*nov, 
                        color=fcolor[icorr], lw=0.5)
        axs[5+icorr].plot((npts*2/3) * np.ones(2), [0, n_sn+nov/2], color='k', lw=0.5)
        axs[5+icorr].text(2/3 + 0.02, 0.5, 'white space start', 
                    transform=axs[5+icorr].transAxes, va='center', rotation=90)
        if icorr==0:
            axs[5].text(1.15, 1.05, 'tuning curves to\nvirtual corridors', 
                        transform=axs[5].transAxes, ha='center')
        axs[5+icorr].set_xlim([0, npts])
        axs[5+icorr].set_ylim([0, n_sn+nov/2])
        axs[5+icorr].spines['left'].set_visible(False)
        axs[5+icorr].set_yticks([])
        axs[5+icorr].set_xticklabels([])
        axs[5+icorr].set_xticks(np.arange(0, 1.25, 0.25))
        axs[5+icorr].set_xlabel('position')