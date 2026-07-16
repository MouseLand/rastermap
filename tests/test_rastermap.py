import inspect

import numpy as np
from rastermap import Rastermap
from rastermap.rastermap import default_settings, sequence_settings, highd_settings


def test_settings_are_valid_kwargs():
    valid = set(inspect.signature(Rastermap.__init__).parameters) - {"self"}
    for settings in (default_settings, sequence_settings, highd_settings):
        unknown = set(settings()) - valid
        assert not unknown, f"{settings.__name__} returns unknown keys: {unknown}"


def test_highd_settings_n_PCs():
    assert highd_settings()["n_PCs"] == 400


def test_rastermap(test_file):
    dat = np.load(test_file)
    spks = dat["spks"]

    model = Rastermap().fit(data=spks)

    assert hasattr(model, "embedding")
    assert hasattr(model, "isort")
    assert hasattr(model, "Usv")
    assert hasattr(model, "Vsv")

def test_rastermap_splits(test_file):
    dat = np.load(test_file)
    spks = dat["spks"]

    model = Rastermap(n_splits=2, n_clusters=20, nc_splits=10).fit(data=spks)

    assert hasattr(model, "embedding")
    assert hasattr(model, "isort")
    assert hasattr(model, "Usv")
    assert hasattr(model, "Vsv")