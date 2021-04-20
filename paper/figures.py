import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import numpy as np
from scipy.stats import zscore

def plot_corridor(S_plot, results, xmin, xmax,
                    ax_events, ax_running, ax_neural):
    axs = [ax_events, ax_running, ax_neural]

    whiteSpcFrameInd = results['whiteSpcStart']
    subRun = results['subRun']
    subRot = results['subRot']
    Image1RewInd = results['Image1Rew']
    Image2RewInd = results['Image2Rew']
    try:
        LickInd = results['Lick']
    except:
        LickInd = results['LickInd']

    VRpos = results['VRpos']
    WallType1 = results['WallType1']
    WallType2 = results['WallType2']
    WallType1_stim = results['WallType1_stim']
    WallType2_stim = results['WallType2_stim']

    h0=axs[0].scatter(WallType1,1.08*np.ones(len(WallType1)), color='r', marker='s')
    h1=axs[0].scatter(WallType2,1.08*np.ones(len(WallType2)), color='b', marker='s')
    h2=axs[0].scatter(Image1RewInd,1.08*np.ones([len(Image1RewInd),]), color='g', marker='^')
    #h2=axs[0].scatter(whiteSpcFrameInd,1.04*np.ones([len(whiteSpcFrameInd),]), color='k', marker='o')
    h3=axs[0].scatter(LickInd,1.01*np.ones([len(LickInd),]), color=[0.5,0.8,0.], marker='.')
    axs[0].axis('off')

    axs[0].legend([h0,h1,h2,h3], ['Context 1 (Reward)', 
                                     'Context 2 (non-Reward)',
                                     'Reward', 
                                     'Licks'
                                     ], loc=(1.04,-3.5))

    axs[1].plot(subRun, color=[0.5, 0.5, 0.5])
    axs[1].set_xlabel('time (frames @ 10Hz)')
    axs[1].spines['left'].set_visible(False)
    axs[1].set_yticks([])
    axs[1].text(xmin, subRun[xmin:xmax].max(),'running speed', color=[0.5, 0.5, 0.5])
    #axs[1].plot(-0.1+subRot, color=[0.8, 0.8, 0.4])


    axs[2].imshow(zscore(S_plot, axis=1), vmin=0, vmax=4, cmap='gray_r', aspect='auto')
    axs[2].axis('off')

    for ax in axs:
        ax.set_xlim([xmin, xmax])

    starts = np.concatenate((np.stack((WallType1, np.zeros(len(WallType1))), axis=1), 
                             np.stack((WallType2, np.ones(len(WallType2))), axis=1)), axis=0)
    starts = starts[starts[:,0].argsort()]
    fcolor = [[1,0,0], [0,0,1]]
    for nn in range(len(starts)):
        start = starts[nn,0]
        icorr = int(starts[nn,1])
        width = whiteSpcFrameInd[nn] - start
        axs[2].add_patch(
                Rectangle(xy=(start, 0), width=width,
                          height=S_plot.shape[0], facecolor=fcolor[icorr], edgecolor=None, alpha=0.2))

    for nn in range(len(Image1RewInd)):
        start = Image1RewInd[nn]
        width = 0
        axs[2].add_patch(
                Rectangle(xy=(start, 0), width=width,
                          height=S_plot.shape[0], facecolor=None, edgecolor='g', alpha=1))
