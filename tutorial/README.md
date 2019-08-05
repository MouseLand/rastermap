# Using rastermap to explore visual cortical activity

## Setting up

First start downloading the data from [figshare](). It will include all the visual cortical responses, the behavioral responses aligned to the neural frames, and a short example video of the mouse's face.

Next we will make an environment with all the packages that we need with the **conda** package manager using the `environment.yml` file:

1. Download the `environment.yml` from this folder *OR*
    1. Clone this repository `git clone https://github.com/MouseLand/rastermap.git` (or pull the latest version if you already have it with `git pull`)
    2. `cd rastermap/tutorial` to be in the same folder with the `environment.yml` folder.
2. Open an anaconda prompt (windows) / command prompt (linux/Mac) with `conda` for **python 3** in the path. In linux/Mac you can check which conda you have with `which conda`, it should be in a subfolder below `anaconda3`.
3. Run `conda env create -f environment.yml`.
4. To activate this new environment, run `conda activate mouseland`.
5. You should see `(mouseland)` on the left side of the terminal line. Now check that you can `python -m suite2p` or `python -m facemap` or `python -m rastermap`.

## MESOSCOPE IN V1

![2pv1](2pv1.JPG)

^ 18,795 neurons in V1 ^

### view the data in [suite2p](https://github.com/MouseLand/suite2p)

Install suite2p and load the data from [figshare]() into suite2p (stat.npy).

### retinotopy

We will now compute the receptive fields, using the [mesoscope1.ipynb](mesoscope1.ipynb). In these experiments we are showing sparse noise stimuli to the mice as they freely run on an air-floating ball.

### explore data using [rastermap](https://github.com/MouseLand/rastermap)

We will use an unsupervised dimensionality reduction technique that works well with neural data. Install rastermap and load the data into the GUI. 

Next we will run rastermap in the notebook so that we can compute receptive fields across the rastermap, open [mesoscope2.ipynb](mesoscope2.ipynb).

What are these neurons doing which don't have clear receptive fields?

### behavioral analysis with [facemap](https://github.com/MouseLand/facemap)

Let's look at what the mouse is doing during the recording. Install facemap using the instructions on the github. Then open the video "cam1_TX39_20Hz.avi" in facemap (this is a subset of the video). You can see how facemap works in [mesoscope3.ipynb](mesoscope3.ipynb).

I've run facemap on the whole movie and aligned them to the neural frames for you. So now let's see how the behavior relates to the neural activity. Open [mesoscope4.ipynb](mesoscope4.ipynb).
