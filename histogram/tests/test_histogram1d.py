from histogram.histogram import Histogram1D
import numpy as np
from copy import copy
import tempfile
import pytest
import os


def _make_dummy_histo():

    bin_edges = np.arange(1000)
    n_histo = 2
    data_shape = (n_histo, )
    histo = Histogram1D(bin_edges=bin_edges, data_shape=data_shape)

    assert histo.shape == data_shape + (len(bin_edges) - 1, )

    return histo


def _make_dummy_2D_1dhisto():

    bin_edges = np.arange(1000)
    n_histo = 2
    data_shape = (n_histo, n_histo)
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

    overflow_total = histo_1.overflow.sum() + histo_2.overflow.sum()
    underflow_total = histo_1.underflow.sum() + histo_2.underflow.sum()

    new_hist = histo_1 + histo_2
    assert new_hist.data.sum() == (histo_1.data.sum() + histo_2.data.sum())
    assert (new_hist.bins == histo_1.bins).all()
    assert (new_hist.bins == histo_2.bins).all()
    assert new_hist.overflow.sum() == overflow_total
    assert new_hist.underflow.sum() == underflow_total


def test_is_equal():

    histo = _make_dummy_histo()
    histo_2 = _make_dummy_histo()

    assert histo == histo_2


def test_is_not_equal():

    histo = _make_dummy_histo()
    histo_2 = _make_dummy_histo()

    histo_2.data += 1

    assert histo != histo_2


@pytest.mark.xfail
def test_is_equal_dummy():

    histo = _make_dummy_histo()

    assert histo == 2
    assert histo != 2


def test_save_and_load():

    histo = _make_dummy_histo()

    for ext in ['.pk', '.fits']:

        with tempfile.NamedTemporaryFile(suffix=ext) as f:

            histo.save(f.name)
            loaded_histo = Histogram1D.load(f.name)

            assert histo == loaded_histo


def test_save_and_load_2D_1dhisto():

    histo = _make_dummy_2D_1dhisto()

    for ext in ['.pk', '.fits']:

        with tempfile.NamedTemporaryFile(suffix=ext) as f:

            histo.save(f.name)
            loaded_histo = Histogram1D.load(f.name)

            assert histo == loaded_histo


def test_load_slice():

    histo = _make_dummy_histo()
    histo[0].data += 1

    for ext in ['.fits']:

        with tempfile.NamedTemporaryFile(suffix=ext) as f:

            histo.save(f.name)
            loaded_histo = Histogram1D.load(f.name, rows=0)
            assert histo[0] == loaded_histo
            assert histo[1] != loaded_histo
            loaded_histo = Histogram1D.load(f.name, rows=(0,))
            assert histo[0] == loaded_histo
            assert histo[1] != loaded_histo

            loaded_histo = Histogram1D.load(f.name, rows=1)
            assert histo[1] == loaded_histo
            assert histo[0] != loaded_histo

            loaded_histo = Histogram1D.load(f.name, rows=(1,))
            assert histo[1] == loaded_histo
            assert histo[0] != loaded_histo


def test_load_slice_index_error():

    histo = _make_dummy_histo()

    with tempfile.NamedTemporaryFile(suffix='.fits') as f:

        histo.save(f.name)

        with pytest.raises(IndexError):

            Histogram1D.load(f.name, rows=(2, ))

        with pytest.raises(IndexError):

            Histogram1D.load(f.name, rows=(-1, ))

        with pytest.raises(IndexError):

            Histogram1D.load(f.name, rows=-1)


def test_load_slice_2D_1dhisto():

    histo = _make_dummy_2D_1dhisto()
    histo[0, 0].data += 1
    histo[1, 1].data += 2

    for ext in ['.fits']:

        with tempfile.NamedTemporaryFile(suffix=ext) as f:

            histo.save(f.name)
            loaded_histo = Histogram1D.load(f.name, rows=(0, 0))
            assert histo[0, 0] == loaded_histo
            assert histo[1, 1] != loaded_histo

            loaded_histo = Histogram1D.load(f.name, rows=(1, 1))
            assert histo[1, 1] == loaded_histo
            assert histo[0, 1] != loaded_histo

            loaded_histo = Histogram1D.load(f.name, rows=(1, ))

            assert histo[1] == loaded_histo
            assert histo[0] != loaded_histo

            loaded_histo = Histogram1D.load(f.name, rows=(0,))

            assert histo[0] == loaded_histo
            assert histo[1] != loaded_histo

            loaded_histo = Histogram1D.load(f.name, rows=(None, 1))

            print(histo[:, 1].data)
            print(loaded_histo.data)

            assert histo[:, 1] == loaded_histo


@pytest.mark.xfail
def test_save_not_defined_format():

    histo = _make_dummy_histo()

    with tempfile.NamedTemporaryFile() as f:

        histo.save(f.name)


def test_mean_and_std():

    histo = _make_dummy_histo()
    n_histo = histo.shape[0]
    bin_edges = histo.bins

    n_data = 50
    datas = np.random.randint(bin_edges.min(), bin_edges.max(),
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


def test_fill():

    histo = _make_dummy_histo()
    n_histo = histo.shape[0]

    data = np.ones((n_histo, 1)) * 50
    n = 100

    for i in range(n):

        histo.fill(data)

    assert (histo.data[:, 50] == n).all()
    assert (histo.data[:, 50 + 1] == 0).all()
    assert (histo.data[:, 50 - 1] == 0).all()


def test_overflow():

    histo = _make_dummy_histo()
    n = 100
    n_histo = histo.shape[0]
    data = np.ones((histo.shape[0], n))

    histo.fill(data * histo.bins.max())
    assert histo.overflow.sum() == (n * n_histo)
    histo.fill(data * histo.bins.max() - 1)
    assert histo.overflow.sum() == (n * n_histo)


def test_underflow():

    histo = _make_dummy_histo()
    n = 100
    n_histo = histo.shape[0]
    data = np.ones((histo.shape[0], n))

    histo.fill(data * histo.bins.min() - 1)
    assert histo.underflow.sum() == (n * n_histo)
    histo.fill(data * histo.bins.min())
    assert histo.underflow.sum() == (n * n_histo)


def test_min_max():

    histo = _make_dummy_histo()

    data = np.arange(histo.shape[0]*10).reshape(histo.shape[0], 10)
    histo.fill(data)

    assert (histo.min() == data.min(axis=-1)).all()
    assert (histo.max() == data.max(axis=-1)).all()


def test_fill_indices():

    bin_edges = np.arange(10)
    n_histo = 4
    n_pixels = 3
    data_shape = (n_histo, n_pixels)
    histo = Histogram1D(bin_edges=bin_edges, data_shape=data_shape)

    index_to_fill = 1
    for i in range(10):

        data = np.ones((n_pixels, 30))
        histo.fill(data, indices=index_to_fill)

    assert histo.data[index_to_fill][:, 1].sum() == 10 * 30 * n_pixels
    assert histo.data.sum() == 10 * 30 * n_pixels


def test_memory_allocation():

    pass


def test_combine():

    histo = _make_dummy_histo()
    sum_histo = histo[0] + histo[1]
    assert sum_histo == histo.combine(axis=0)


def test_simple_import():

    from histogram import Histogram1D


def test_write_info_empty_histo():

    histo = _make_dummy_histo()
    assert histo[0].is_empty()
    histo._write_info(index=0)


def test_save_figures_empty_histo():

    histo = _make_dummy_histo()
    assert histo.is_empty()

    with tempfile.NamedTemporaryFile(suffix='.pdf') as f:

        histo.save_figures(f.name)
        assert os.path.exists(f.name)


def test_save_figures_histo():

    histo = _make_dummy_histo()

    for i in range(100):
        data = np.random.normal(500, 10, size=(2, 200))
        histo.fill(data)

    with tempfile.NamedTemporaryFile(suffix='.pdf') as f:

        histo.save_figures(f.name)
        assert os.path.exists(f.name)

    histo2d = _make_dummy_2D_1dhisto()

    for i in range(100):
        data = np.random.normal(500, 10, size=(2, 2, 200))
        histo2d.fill(data)

    with tempfile.NamedTemporaryFile(suffix='.pdf') as f:

        histo2d.save_figures(f.name)
        assert os.path.exists(f.name)


if __name__ == '__main__':

    test_fill()
    test_fill_indices()
    test_load_slice()
    test_load_slice_index_error()
    histo2d = _make_dummy_2D_1dhisto()

    for i in range(100):
        data = np.random.normal(500, 10, size=(2, 2, 200))
        histo2d.fill(data)

    histo2d.save_figures('test.pdf')
