"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os, warnings
import numpy as np
import scipy.io as sio
from scipy.stats import zscore

def _load_dict(dat, keys):
    X, Usv, Vsv, xpos, ypos, xy = None, None, None, None, None, None
    other_keys = []
    for key in keys:
        if key=="Usv":
            Usv = dat["Usv"]
        elif key=="Vsv":
            Vsv = dat["Vsv"]
        elif key=="U":
            U = dat["U"]
        elif key=="U0":
            U = dat["U0"]
        elif key=="V":
            V = dat["V"]
        elif key=="V0":
            V = dat["V0"]
        elif key=="Sv":
            Sv = dat["Sv"]
        elif key=="sv":
            Sv = dat["sv"]
        elif key=="X":
            X = dat["X"]
        elif key=="spks":
            X = dat["spks"]
        elif key=="xpos":
            xpos = dat["xpos"]
        elif key=="ypos":
            ypos = dat["ypos"]
        elif key=="xy":
            xy = dat["xy"]
        elif key=="xyz":
            xy = dat["xyz"]
        else:
            other_keys.append(key)

    if X is None:
        if Usv is None and U is not None and Sv is None:
            if Vsv is not None:
                Sv = (Vsv**2).sum(axis=0)**0.5
            else:
                raise ValueError("no Sv scaling for PCs available")
        elif Vsv is None and V is not None and Sv is None:
            if Usv is not None:
                Sv = (Usv**2).sum(axis=0)**0.5
            else:
                raise ValueError("no Sv scaling for PCs available")
        if Usv is None and U is not None and Sv is not None:
            Usv = U * Sv
        if Vsv is None and V is not None and Sv is not None:
            Vsv = V * Sv

        if Usv.shape[-1] != Vsv.shape[-1]:
            raise ValueError("Usv and Vsv must have the same number of components")
        if Usv.ndim > 3:
            raise ValueError("Usv cannot have more than 3 dimensions")
        if Vsv.ndim != 2:
            raise ValueError("Vsv must have 2 dimensions")

    if xpos is not None and xy is None:
        xy = np.stack((ypos, xpos), axis=1)

    if xy is not None:
        if xy.ndim != 2:
            print("cannot use xy from file: x and y positions of neurons must be 2-dimensional")
            xy = None
        elif xy.shape[0]==2 or xy.shape[0]==3:
            xy = xy.T
        if xy is not None:
            if X is not None and X.shape[0]!=xy.shape[0]:
                xy = None
            elif Usv is not None and Usv.shape[0]!=xy.shape[0]:
                xy = None
            if xy is None:
                print("cannot use xy from file: x and y positions of neurons are not same size as activity")

    return X, Usv, Vsv, xy

def load_activity(filename):
    ext = os.path.splitext(filename)[-1]
    print("Loading " + filename)
    Usv, Vsv = None, None
    if ext == ".mat":
        try:
            X = sio.loadmat(filename)
            if isinstance(X, dict):
                for i, key in enumerate(X.keys()):
                    if key not in ["__header__", "__version__", "__globals__"]:
                        X = X[key]
        except NotImplementedError:
            try:
                import mat73
            except ImportError:
                print("please 'pip install mat73'")
            X = mat73.loadmat(filename)
            if isinstance(X, dict):
                keys = []
                for i, key in enumerate(X.keys()):
                    if key not in ["__header__", "__version__", "__globals__"]:
                        keys.append(key)
                X, Usv, Vsv, xy = _load_dict(X, keys)
    elif ext == ".npy":
        X = np.load(filename, allow_pickle=True)
        if isinstance(X.item(), dict):
            dat = X.item()
            keys = dat.keys()
            X, Usv, Vsv, xy = _load_dict(dat, keys)
    elif ext == ".npz":
        dat = np.load(filename, allow_pickle=True)
        keys = dat.files
        X, Usv, Vsv, xy = _load_dict(dat, keys)    
    elif ext == ".nwb":
        X, xy = _load_nwb(filename)
    else:
        raise Exception("Invalid file type")

    if X is None and (Usv is None or Vsv is None):
        return
    if X is not None:
        if X.ndim == 1:
            raise ValueError(
                "ERROR: 1D array provided, but rastermap requires 2D array"
            )
        elif X.ndim > 3:
            raise ValueError(
                "ERROR: nD array provided (n>3), but rastermap requires 2D array"
            )
        elif X.ndim == 3:
            warnings.warn(
                "WARNING: 3D array provided (n>3), rastermap requires 2D array, will flatten to 2D"
            )
        if X.shape[0] < 10:
            raise ValueError(
                "ERROR: matrix with fewer than 10 neurons provided"
            )

        if len(X.shape) == 3:
            print(
                f"activity matrix has third dimension of size {X.shape[-1]}, flattening matrix to size ({X.shape[0]}, {X.shape[1] * X.shape[-1]}"
            )
            X = X.reshape(X.shape[0], -1)
    
    return X, Usv, Vsv, xy


def _cell_center(voxel_mask):
    x = np.median(np.array([v[0] for v in voxel_mask]))
    y = np.median(np.array([v[1] for v in voxel_mask]))
    return np.array([x, y])

def _load_nwb(filename):
    try:
        from pynwb import NWBHDF5IO, NWBFile, TimeSeries
        from pynwb.ophys import (
            DfOverF,
            Fluorescence  ,
            RoiResponseSeries      
        )
    except:
        raise ImportError("pynwb not installed, please pip install pynwb")
    """ load ophys data from nwb"""
    with NWBHDF5IO(filename, "r") as io:
        read_nwbfile = io.read()

        # load neural activity
        X = [x for x in read_nwbfile.objects.values() if isinstance(x, Fluorescence)]
        names = [x.name for x in read_nwbfile.objects.values() if isinstance(x, Fluorescence)]
        if len(X) == 0:
            X = [x for x in read_nwbfile.objects.values() if isinstance(x, DfOverF)]
            names = [x.name for x in read_nwbfile.objects.values() if isinstance(x, DfOverF)]
            if len(X) == 0:
                X = [x for x in read_nwbfile.objects.values() if isinstance(x, RoiResponseSeries)]
                names = [x.name for x in read_nwbfile.objects.values() if isinstance(x, RoiResponseSeries)]
                
            
        if len(X) > 0:
            if len(X) == 3 and "Deconvolved" in names:
                X = X[names.index("Deconvolved")]
            elif len(X) > 1:
                # todo: allow user to select series
                print(f"more than one series to choose from, taking first series {names[0]}")
                X = X[0]
            elif len(X) == 1:
                X = X[0]

            planes = list(X.roi_response_series.keys())

            spks = np.concatenate(([X[plane].data[:] for plane in planes]), 
                                    axis=1).T 
            spks = spks.astype("float32")
            ids = np.concatenate(([X[plane].rois.data[:] for plane in planes]), 
                                    axis=0)
            
            if hasattr(X[planes[0]].rois[0], "image_mask"):
                roikey = "image_mask"
            elif hasattr(X[planes[0]].rois[0], "voxel_mask"):
                roikey = "voxel_mask"
            else:
                roikey = None

            if roikey is not None:
                xy = np.concatenate([np.array([_cell_center(roi[roikey].values[0]) 
                                            for roi in X[plane].rois]) 
                                                for plane in planes])
            else:
                voxel_masks = np.concatenate([np.array([roi
                                        for roi in X[plane].rois]) 
                                            for plane in planes])
                xy = np.stack([_cell_center(vm[0][0]) for vm in voxel_masks], 
                                        axis=0)
        else:
            raise ValueError("not an ophys NWB file with a Fluorescence or DfOverF roi_response_series")

    return spks, xy
        

def _load_iscell(filename):
    basename = os.path.split(filename)[0]
    try:
        file_iscell = os.path.join(basename, "iscell.npy")
        iscell = np.load(file_iscell)
        probcell = iscell[:, 1]
        iscell = iscell[:, 0].astype("bool")
    except (ValueError, OSError, RuntimeError, TypeError, NameError):
        iscell = None
        file_iscell = None
    return iscell, file_iscell

def _load_stat(filename):
    basename = os.path.split(filename)[0]
    try:
        file_stat = os.path.join(basename, "stat.npy")
        stat = np.load(file_stat, allow_pickle=True)
        xy = np.array([s["med"] for s in stat])
    except (ValueError, OSError, RuntimeError, TypeError, NameError):
        xy = None
        file_stat = None
    return xy, file_stat
