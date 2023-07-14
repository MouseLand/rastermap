import numpy as np
from rastermap import Rastermap


def test_rastermap(test_file):
    dat = np.load(test_file)
    spks = dat["spks"]

    model = Rastermap().fit(data=spks)

    assert hasattr(model, "embedding")
    assert hasattr(model, "isort")
    assert hasattr(model, "Usv")
    assert hasattr(model, "Vsv")