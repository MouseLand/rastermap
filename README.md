# rastermap

This algorithm computes a 1D or 2D continuous sorting of neural activity. It assumes that the spike matrix `S` is neurons by timepoints. We have a python 3 and a matlab implementation. See the [demos](rastermap/demos/) for jupyter notebooks using it, and some example data.

Here is what the output looks like for a segment of a mesoscope recording (2.5Hz sampling rate) (sorted in neural space, but not time space):

![rastersorted](example.png)

Here is what the output looks like for a recording of 64,000 neurons in a larval zebrafish (data [here](https://figshare.com/articles/Whole-brain_light-sheet_imaging_data/7272617/1), thanks to Chen, Mu, Hu, Kuan et al for sharing!). The upper left plot is the 2D embedding with boxes around clusters (which the user draws in the GUI). The plot on the right shows the activity of the clusters. The lower left plot is a Z-stack of the neurons in the tissue, colored according to their 2D position in the Rastermap embedding.

![fishbrain](fishGUI3.png)

## Installation

You can just download the github folder as outlined above or you can pip install the package:
```
pip install rastermap
```

## Using (python) rastermap

rastermap can be run the same way as the T-SNE embedding algorithm or other algorithms in scikit-learn. **RMAP** is a class which has functions *fit*, *fit_transform*, and *transform* (embeds new points into original embedding).

**(input should be n_samples x n_features like t-sne, etc)**

```
# >> from github <<
import sys
sys.path.append('/media/carsen/DATA2/Github/rastermap/rastermap/')
from mapping import RMAP

# >> from pip <<
from rastermap import RMAP

model = RMAP(n_components=1, n_X=30, n_Y=100, iPC=np.arange(0,200).astype(np.int32), init='random')

# fit does not return anything, it adds attributes to model
# attributes: embedding, isort1, isort2

model.fit(sp)
plt.imshow(sp[model.isort1, model.isort2])

# fit_transform returns embedding (upsampled cluster identities)
embedding = model.fit_transform(sp)

# transform can be used on new samples with the same number of features as sp
embed2 = model.transform(sp2)
```

## Parameters

Rastermap first takes the specified PCs of the data, and then embeds them into n_X clusters. It returns upsampled cluster identities (n_X x upsamp). Clusters are also computed across Y (n_Y) and smoothed, to help with fitting.

- **n_components** : int, optional (default: 1)
        dimension of the embedding space
- **n_X** : int, optional (default: 30)
        number of clusters in X
- **n_Y** :  int, optional (default: 100)
        number of clusters in Y: will be used to smooth data before sorting in X
- **iPC**  : nparray, int, optional (default: 0-199)
        which PCs to use during optimization
- **upsamp** : int, optional (default: 25)
        embedding is upsampled in last iteration using kriging interpolation
- **sig_upsamp** : float, optional (default: 1.0)
        stddev of Gaussian in kriging interpolation for upsampled estimation
- **sig_Y** : float, optional (default: 3.0)
        stddev of Gaussian smoothing in Y before sorting in X
- **sig_anneal**: 1D float array, optional (default: starts at 6.0, decreases to 1.0)
        stddev of Gaussian smoothing of clusters, changes across iterations
        default is 50 iterations (last 20 at 1.0)
- **init** : initialization of algorithm (default: 'pca')
        can use 'pca', 'random', or a matrix n_samples x n_components
        
## Outputs

RMAP model has the following attributes after running 'fit':
- **embedding** : array-like, shape (n_samples, n_components)
        Stores the embedding vectors.
- **u,sv,v** : singular value decomposition of data S, potentially with smoothing
- **isort1** : sorting along first dimension (n_samples) of matrix
- **isort2** : sorting along second dimension (n_features) of matrix (if n_Y > 0)


## Requirements

This package was written for Python 3 and relies on **numpy** and **scipy**. The Python3.x Anaconda distributions will contain all the dependencies.

## Matlab

The matlab version requires Matlab 2016a or later. If you want to use the GPU acceleration (useGPU=1), then you need an NVIDIA GPU and the Parallel Computing Toolbox. Otherwise, I don't think it requires any additional toolboxes, but please let me know if it does in the issues.

The matlab code needs to be cleaned up but the main function to call is `mapTmap.m`. This function is used in the example script `loadFromPython.m` (loads suite2p outputs, requires [npy-matlab](https://github.com/kwikteam/npy-matlab)).
