from histogram.histogram import Histogram1D
import numpy as np
from copy import copy
import tempfile


def _make_dummy_histo():

    bin_edges = np.arange(1000)
    n_histo = 100
    data_shape = (n_histo,)
    histo = Histogram1D(bin_edges=bin_edges, data_shape=data_shape)

    assert histo.shape == data_shape + (len(bin_edges) - 1, )

    return histo


def test_init():

    histo = _make_dummy_histo()

    assert histo.data.sum() == 0
    assert histo.underflow.sum() == 0
    assert histo.overflow.sum() == 0


def test_add():

    histo = _make_dummy_histo()

    n_histo = histo.shape[0]

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

    histo = _make_dummy_histo()

    with tempfile.NamedTemporaryFile() as f:

        histo.save(f.name)
        Histogram1D.load(f.name)


def test_mean_and_std():

    histo = _make_dummy_histo()
    n_histo = histo.shape[0]
    bin_edges = histo.bins

    n_data = 50
    datas = np.random.randint(bin_edges.min()+1, bin_edges.max()-1,
                              size=(n_data, n_histo, 30))

    mean_data = np.zeros(n_histo)
    std_data = np.zeros(n_histo)

    for data in datas:

        mean_data += np.mean(data, axis=-1)
        std_data += np.mean(data**2, axis=-1)
        histo.fill(data)

    mean_data /= n_data
    std_data /= n_data
    std_data -= mean_data**2
    std_data = np.sqrt(std_data)

    mean_histo = histo.mean()
    std_histo = histo.std()

    assert mean_histo.shape == (n_histo, )
    assert (mean_histo - mean_data).sum() <= 1e-11
    assert (std_histo - std_data).sum() <= 1e-11
