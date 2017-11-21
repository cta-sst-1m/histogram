from histogram.histogram import Histogram1D
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    my_histo = Histogram1D(data_shape=(1296, ), bin_edges=np.arange(0, 4096, 1), axis_name='ADC')
    dat = np.random.normal(200, 10, size=(1296, 800))
    my_histo.fill(data_points=dat)
    dat = np.random.normal(300, 10, size=(1296, 1000))
    my_histo.fill(data_points=dat)

    print(np.mean(dat, axis=-1))

    print(my_histo.bin_centers)
    print(my_histo.data)
    print(my_histo.underflow)
    print(my_histo.overflow)
    print(my_histo.mean())
    print(my_histo.std())

    my_histo.show(index=(10, ), normed=False)
    plt.show()