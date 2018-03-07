from histogram.histogram import Histogram1D
import numpy as np
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':

    n_events, n_pixels, n_samples = 1000, 1296, 50
    my_histo = Histogram1D(
                           data_shape=(n_pixels, ),
                           bin_edges=np.arange(-5, 5, 0.2),
                           axis_name='ADC')
    dat = np.random.normal(size=(n_events, n_pixels, n_samples))

    t_0 = time.time()
    for data in dat:

        my_histo.fill(data_points=data)

    print('The Histogram1D took {} seconds to be filled'.format(
        time.time() - t_0))

    t_0 = time.time()
    for data in dat:

        hists = np.apply_along_axis(
            np.histogram,
            axis=1,
            arr=data,
            range=(-5, 5),
            bins=50)

    print('Using numpy np.apply_along_axis took {} seconds'.format(
        time.time() - t_0))

    my_histo.draw(index=(10, ), normed=False)

    my_histo.draw_all()

    plt.show()