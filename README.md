# Rastermap

![tests](https://github.com/mouseland/rastermap/actions/workflows/test_and_deploy.yml/badge.svg)
[![codecov](https://codecov.io/gh/MouseLand/rastermap/branch/main/graph/badge.svg?token=9FFo4zNtYP)](https://codecov.io/gh/MouseLand/rastermap)
[![PyPI version](https://badge.fury.io/py/rastermap.svg)](https://badge.fury.io/py/rastermap)
[![Downloads](https://pepy.tech/badge/rastermap)](https://pepy.tech/project/rastermap)
[![Downloads](https://pepy.tech/badge/rastermap/month)](https://pepy.tech/project/rastermap)
[![Python version](https://img.shields.io/pypi/pyversions/rastermap)](https://pypistats.org/packages/rastermap)
[![Licence: GPL v3](https://img.shields.io/github/license/MouseLand/rastermap)](https://github.com/MouseLand/rastermap/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/MouseLand/rastermap)](https://github.com/MouseLand/rastermap/graphs/contributors)
[![repo size](https://img.shields.io/github/repo-size/MouseLand/rastermap)](https://github.com/MouseLand/rastermap/)
[![GitHub stars](https://img.shields.io/github/stars/MouseLand/rastermap?style=social)](https://github.com/MouseLand/rastermap/)
[![GitHub forks](https://img.shields.io/github/forks/MouseLand/rastermap?style=social)](https://github.com/MouseLand/rastermap/)


Rastermap is a discovry algorithm for neural data. The algorithm was written by 
Carsen Stringer and Marius Pachitariu. To learn about Rastermap, read the [paper]() or watch the [talk](). For support,  please open an [issue](https://github.com/MouseLand/rastermap/issues). Please see install instructions [below](README.md/#Installation).

Rastermap runs in python 3.8+ and has a graphical user interface (GUI) for running it easily. Rastermap can also be run in a jupyter notebook locally or on google colab:
* [run_rastermap_largescale.ipynb](notebooks/run_rastermap_largescale.ipynb) notebook shows how to use it with large-scale data (> 200 neurons)
* [run_rastermap.ipynb](notebooks/run_rastermap.ipynb) notebook shows how to use it with small to medium sized data (< 200 neurons)
* [tutorial.ipynb](notebooks/tutorial.ipynb) is a guided tutorial for integrating rastermap and facemap in an attempt to understand behavioral representations

Here is what the output looks like for a segment of a mesoscope recording in a mouse during spontaneous activity (3.2Hz sampling rate), compared to random neural sorting:

<img src="https://www.suite2p.org/static/images/example_sorting_spont.png" width="600" alt="random sorting and rastermap sorting of spontaneous activity"/>

Here is what the output looks like for a recording of 64,000 neurons in a larval zebrafish (data [here](https://figshare.com/articles/Whole-brain_light-sheet_imaging_data/7272617/1), thanks to Chen, Mu, Hu, Kuan et al / Ahrens lab for sharing). The plot on the left shows the sorted activity, and the right plot is the 2D positions of the neurons in the tissue, divided into 18 clusters according to their 1D position in the Rastermap embedding:

<img src="https://www.suite2p.org/static/images/rastermap_zebrafish.png" width="800" alt="wholebrain neural activity from a zebrafish sorted by rastermap"/>

# Installation

## Local installation (< 2 minutes)

### System requirements

Linux, Windows and Mac OS are supported for running the code. For running the graphical interface you will need a Mac OS later than Yosemite. At least 8GB of RAM is required to run the software. 16GB-32GB may be required for larger images and 3D volumes. The software has been heavily tested on Windows 10 and Ubuntu 20.04 and less well-tested on Mac OS. Please open an [issue](https://github.com/MouseLand/rastermap/issues) if you have problems with installation.

### Instructions

Recommended to install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.x** and your operating system. Note you might need to use an anaconda prompt (windows) if you did not add anaconda to the path. Open an anaconda prompt / command prompt with **python 3** in the path, then:

~~~sh
pip install rastermap
~~~

For the GUI 
~~~sh
pip install rastermap[gui]
~~~

Rastermap has only a few dependencies so you may not need to make a special environment for it 
(e.g. it should work in a `suite2p` or `facemap` environment), but if the pip install above does not work,
 please follow these instructions:

1. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path.
2. Create a new environment with `conda create --name rastermap python=3.8`. Python 3.9 and 3.10 will likely work fine as well.
4. To activate this new environment, run `conda activate rastermap`
5. To install the minimal version of rastermap, run `pip install rastermap`.  
6. To install rastermap and the GUI, run `pip install rastermap[gui]`. If you're on a zsh server, you may need to use ' ' around the rastermap[gui] call: `pip install 'rastermap[gui]'`.

To upgrade rastermap (package [here](https://pypi.org/project/rastermap/)), run the following in the environment:

~~~sh
pip install rastermap --upgrade
~~~

If you have an older `rastermap` environment you can remove it with `conda env remove -n rastermap` before creating a new one.

Note you will always have to run **conda activate rastermap** before you run rastermap. If you want to run jupyter notebooks in this environment, then also `pip install notebook`.

### Dependencies

This package relies on the awesomeness of **numpy**, **scipy**, **numba**, **scikit-learn**, **PyQt5**, **PyQt5.sip** and **pyqtgraph**. You can pip install or conda install all of these packages. If having issues with **PyQt5**, then make an Anaconda environment and try to install within it with `pip install PyQt5` or `conda install pyqt`.

# Using rastermap

## GUI

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path. Then run:

~~~sh
python -m rastermap
~~~

To start using the GUI, save your data into an npy file that is just a matrix that is neurons x timepoints. Then "File > Load data matrix" and choose this file. Next click "Run > Run rastermap" and click run. See the parameters section to learn about the parameters.

The GUI will start with a highlighted region that you can drag to visualize the average activity of neurons in a given part of the plot. To draw more regions, you right-click to start a region, then right-click to end it. The neurons' activity traces then show up on the botton of the GUI, and if the neuron positions are loaded, you will see them colored by the region color. You can delete a region by holding CTRL and clicking on it. You can save the ROIs you've drawn with the "Save > Save processed data" button. They will save along with the embedding so you can reload the file with the "Load processed data" option.

NOTE: If you are using suite2p "spks.npy", then the GUI will automatically use the "iscell.npy" file in the same folder to subsample your recording with the chosen neurons, and will automatically load 
the neuron positions from the "stat.npy" file.

## In a notebook

For this, `pip install notebook` and `pip install matpltolib`.

See example notebooks for more details: [run_rastermap_largescale.ipynb](notebooks/run_rastermap_largescale.ipynb), [run_rastermap.ipynb](notebooks/run_rastermap.ipynb), and [tutorial.ipynb](notebooks/tutorial.ipynb).

Short code snippet:

```
import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap, utils
from scipy.stats import zscore

# spks is neurons by time
spks = np.load("spks.npy").astype("float32")
spks = zscore(spks, axis=1)

# fit rastermap
model = Rastermap(n_PCs=200, n_clusters=100, 
                  locality=0.75, time_lag_window=5).fit(spks)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
X_embedding = zscore(utils.bin1d(spks, bin_size=25, axis=0), axis=1)

# plot
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
```

## From the command line

Save an "ops.npy" file with the parameters and a "spks.npy" file with a matrix of neurons by time, and run

~~~sh
python -m rastermap --S spks.npy --ops ops.npy
~~~

# Parameters

TBD

# Outputs

TBD

# License

Copyright (C) 2023 Howard Hughes Medical Institute Janelia Research Campus, the labs of Carsen Stringer and Marius Pachitariu.

**This code is licensed under GPL v3 (no redistribution without credit, and no redistribution in private repos, see the [license](LICENSE) for more details).**