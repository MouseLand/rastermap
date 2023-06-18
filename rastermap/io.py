"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os, warnings
import numpy as np
import scipy.io as sio
from scipy.stats import zscore

def _load_dict(dat, keys):
    X, Usv, Vsv = None, None, None
    other_keys = []
    for key in keys:
        if key=="Usv":
            Usv = dat["Usv"]
        elif key=="Vsv":
            Vsv = dat["Vsv"]
        elif key=="U":
            U = dat["U"]
        elif key=="V":
            V = dat["V"]
        elif key=="Sv":
            Sv = dat["Sv"]
        elif key=="X":
            X = dat["X"]
        elif key=="spks":
            X = dat["spks"]
        else:
            other_keys.append(key)

    if Usv is None and U is not None and Sv is not None:
        Usv = U * Sv
    if Vsv is None and V is not None and Sv is not None:
        Vsv = V * Sv

    if X is None and len(other_keys) > 0:
        X = dat[other_keys[0]]

    if Usv is not None and Vsv is not None:
        if Usv.shape[-1] != Vsv.shape[-1]:
            raise ValueError("Usv and Vsv must have the same number of components")
        if Usv.ndim > 3:
            raise ValueError("Usv cannot have more than 3 dimensions")
        if Vsv.ndim != 2:
            raise ValueError("Vsv must have 2 dimensions")

    return X, Usv, Vsv

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
                X, Usv, Vsv = _load_dict(X, keys)
    elif ext == ".npy":
        X = np.load(filename, allow_pickle=True)
        if isinstance(X, dict):
            dat = X.item()
            keys = dat.keys()
            X, Usv, Vsv = _load_dict(dat, keys)
    elif ext == ".npz":
        dat = np.load(filename, allow_pickle=True)
        keys = dat.files
        X, Usv, Vsv = _load_dict(dat, keys)    
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
    
    return X, Usv, Vsv

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
