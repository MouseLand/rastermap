"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import os, tempfile, shutil
from urllib.request import urlopen
from tqdm import tqdm
from pathlib import Path

def bin1d(X, bin_size, axis=0):
    """ mean bin over axis of data with bin bin_size """
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        Xb = Xb[:size[axis]//bin_size*bin_size].reshape((size[axis]//bin_size, bin_size, -1)).mean(axis=1)
        Xb = Xb.swapaxes(axis, 0)
        size[axis] = Xb.shape[axis]
        Xb = Xb.reshape(size)
        return Xb
    else:
        return X

def split_traintest(n_t, frac=0.25, n_segs=20, pad=3, split_time=False):
    """this returns deterministic split of train and test in time chunks
    
    Parameters
    ----------
    n_t : int
        number of timepoints to split
    frac : float (optional, default 0.25)
        fraction of points to put in test set
    pad : int (optional, default 3)
        number of timepoints to exclude from test set before and after training segment
    split_time : bool (optional, default False)
        split train and test into beginning and end of experiment
    Returns
    --------
    itrain: 2D int array
        times in train set, arranged in chunks
    
    itest: 2D int array
        times in test set, arranged in chunks
    """
    #usu want 20 segs, but might not have enough frames for that
    n_segs = int(min(n_segs, n_t/4)) 
    n_len = int(np.floor(n_t/n_segs))
    inds_train = np.linspace(0, n_t - n_len - 5, n_segs).astype(int)
    if not split_time:
        l_train = int(np.floor(n_len * (1-frac)))
        inds_test = inds_train + l_train + pad
        l_test = np.diff(np.stack((inds_train, inds_train + l_train)).T.flatten()).min() - pad
    else:
        inds_test = inds_train[:int(np.floor(n_segs*frac))]
        inds_train = inds_train[int(np.floor(n_segs*frac)):]
        l_train = n_len - 10
        l_test = l_train
    itrain = (inds_train[:,np.newaxis] + np.arange(0, l_train, 1, int))
    itest = (inds_test[:,np.newaxis] + np.arange(0, l_test, 1, int))
    return itrain, itest

def download_url_to_file(url, dst, progress=True):
    r"""Download object at the given URL to a local path.
            Thanks to torch, slightly modified
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def download_data(data_type="hippocampus"):
    if data_type=="widefield":
        url = "https://osf.io/5d8q7/download"
    elif data_type=="spont2":
        url = "https://osf.io/8xg7n/download"
    elif data_type=="hippocampus":
        url = "https://osf.io/szmw6/download"
    elif data_type=="fish":
        url = "https://osf.io/2w8pa/download"
    ddir = Path.home().joinpath('.rastermap')
    ddir.mkdir(exist_ok=True)
    data_dir = ddir.joinpath('data')
    data_dir.mkdir(exist_ok=True)
    data_file = str(data_dir.joinpath(f"{data_type}_data.npz"))
    if not os.path.exists(data_file):
        download_url_to_file(url, data_file)
    return data_file
