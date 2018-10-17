from histogram.histogram import Histogram1D
import numpy as np
from copy import copy
import tempfile

bin_edges = np.arange(1000)
n_histo = 100
data_shape = (n_histo,)
histo = Histogram1D(bin_edges=bin_edges, data_shape=data_shape)


def test_create():

    assert histo.shape == data_shape + (len(bin_edges) - 1, )
    assert histo.data.sum() == 0
    assert histo.underflow.sum() == 0
    assert histo.overflow.sum() == 0


def test_add():

    histo_1 = copy(histo)
    histo_2 = copy(histo)

    for i in range(100):

        data = np.random.normal(500, 10, size=(n_histo, 200))
        histo_1.fill(data)
        histo_2.fill(data)

    new_hist = histo_1 + histo_2
    assert new_hist.data.sum() == (histo_1.data.sum() + histo_2.data.sum())
    assert (new_hist.bins == histo_1.bins).all()
    assert (new_hist.bins == histo_2.bins).all()
    assert new_hist.overflow.sum() == histo_1.overflow.sum() + histo_2.overflow.sum()
    assert new_hist.underflow.sum() == histo_1.underflow.sum() + histo_2.underflow.sum()


def test_save_and_load():

    with tempfile.NamedTemporaryFile() as f:

        histo.save(f.name)
        Histogram1D.load(f.name)
