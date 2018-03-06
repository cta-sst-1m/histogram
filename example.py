from histogram.histogram import Histogram1D
import numpy as np
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':

    my_histo = Histogram1D(data_shape=(2048, ), bin_edges=np.arange(-5, 5, 0.2), axis_name='ADC')
    dat = np.random.normal(size=(2048, 10000))

    t_0 = time.time()
    my_histo.fill(data_points=dat)
    print(time.time() - t_0)

    t_0 = time.time()
    hists = np.apply_along_axis(np.histogram, axis=1, arr=dat, range=(-5, 5), bins=50)
    print(time.time() - t_0)

    # dat = np.random.normal(300, 10, size=(1296, 10000))
    # my_histo.fill(data_points=dat)

    print(np.mean(dat, axis=-1))

    print(my_histo._bin_centers())
    print(my_histo.data)
    print(my_histo.underflow)
    print(my_histo.overflow)
    print(my_histo.mean())
    print(my_histo.std())

    my_histo.draw(index=(10, ), normed=False)
    plt.show()