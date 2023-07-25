import os, warnings, time, tempfile, datetime, pathlib, shutil, subprocess
from urllib.request import urlopen
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
import pytest

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

@pytest.fixture()
def test_file():
    ddir = Path.home().joinpath('.rastermap')
    ddir.mkdir(exist_ok=True)
    data_dir = ddir.joinpath('data')
    data_dir.mkdir(exist_ok=True)
    url = "http://www.suite2p.org/static/test_data/neuropop_test_data.npz"
    test_file = str(data_dir.joinpath("neuropop_test_data.npz"))
    if not os.path.exists(test_file):
        download_url_to_file(url, test_file)
    return test_file