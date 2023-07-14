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


Rastermap is a visualization algorithm for neural data. The algorithm was written by 
Carsen Stringer and Marius Pachitariu. To learn about Rastermap, read the [paper]() 
or watch the [talk](). For support,  please open an [issue](https://github.com/MouseLand/rastermap/issues). Please see install instructions [below](README.md/#Installation).

It assumes that the data matrix `data` is neurons (or voxels) by timepoints. Rastermap runs in python 3.8+ and has a GUI for running it easily. See the [run_rastermap.ipynb](notebooks/run_rastermap.ipynb) notebook to see an example for how to use it, it includes a download for example data. We also have a guided [tutorial.ipynb](notebooks/tutorial.ipynb) integrating rastermap and facemap in an attempt to understand behavioral representations.

Here is what the output looks like for a segment of a mesoscope recording (3.2Hz sampling rate) (sorted in neural space):

TBD

Here is what the output looks like for a recording of 64,000 neurons in a larval zebrafish (data [here](https://figshare.com/articles/Whole-brain_light-sheet_imaging_data/7272617/1), thanks to Chen, Mu, Hu, Kuan et al / Ahrens lab for sharing!). The plot on the left shows the activity of the clusters. The right plot is the positions neurons in the tissue, colored according to their 1D position in the Rastermap embedding.

TBD

# Installation

## Local installation (< 2 minutes)

### System requirements

Linux, Windows and Mac OS are supported for running the code. For running the graphical interface you will need a Mac OS later than Yosemite. At least 8GB of RAM is required to run the software. 16GB-32GB may be required for larger images and 3D volumes. The software has been heavily tested on Windows 10 and Ubuntu 18.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

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
(it should work in a `suite2p` or `facemap` environment), but if the pip install above does not work,
 please follow these instructions:

1. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path.
2. Create a new environment with `conda create --name rastermap python=3.8`. Python 3.9 and 3.10 will likely work as well.
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

This package relies on the awesomeness of **numpy**, **scipy**, **numba**, **scikit-learn**, **PyQt5**, **PyQt5.sip** and **pyqtgraph**. You can pip install or conda install all of these packages. If having issues with **PyQt5**, then make an Anaconda environment and try to install within it.

# Using rastermap

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path. Then run:

~~~sh
python -m rastermap
~~~

To starting using the GUI, save your data into an npy file that is just a matrix that is neurons x timepoints. Then "File > Load data matrix" and choose this file. Next click "Run > Run rastermap" and click run. See the parameters section to learn about the parameters.

The GUI will start with a highlighted region that you can drag to visualize the average activity of neurons in a given part of the plot. To draw more regions, you right-click to start a region, then right-click to end it. The neurons' activity traces then show up on the botton of the GUI, and if the neuron positions are loaded, you will see them colored by the region color. You can delete a region by holding CTRL and clicking on it. You can save the ROIs you've drawn with the "Save > Save processed data" button. They will save along with the embedding so you can reload the file with the "Load processed data" option.

```
import numpy as np
model = np.load('embedding.npy')
model = model.dict()
y = model['embedding'] # neurons x 1
```

NOTE: If you are using suite2p "spks.npy", then the GUI will automatically use the "iscell.npy" file in the same folder to subsample your recording with the chosen neurons, and will automatically load 
the neuron positions from the "stat.npy" file.

## In a notebook

Please see example notebooks [run_rastermap.ipynb](notebooks/run_rastermap.ipynb) and [tutorial.ipynb](notebooks/tutorial.ipynb).

## From the command line

TBD

# Parameters

TBD

# Outputs

TBD

# License

Copyright (C) 2023 Howard Hughes Medical Institute Janelia Research Campus, the labs of Carsen Stringer and Marius Pachitariu.

**This code is licensed under GPL v3 (no redistribution without credit, and no redistribution in private repos, see the [license](LICENSE) for more details).**