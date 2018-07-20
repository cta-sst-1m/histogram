from histogram.histogram import Histogram1D
from histogram.fit import HistogramFitter
import numpy as np
import matplotlib.pyplot as plt
import time


class MyHistogramFitter(HistogramFitter):

    def initialize_fit(self):

        x = self.bin_centers
        y = self.count

        mean = np.average(x, weights=y)
        std = np.average((x - mean)**2, weights=y)
        std = np.sqrt(std)
        amplitude = np.sum(y, dtype=np.float)

        self.initial_parameters = {'mean': mean, 'std': std,
                                   'amplitude': amplitude}

    def compute_fit_boundaries(self):

        bounds = {}

        for key, val in self.initial_parameters.items():

            if val > 0:
                bounds['limit_'+key] = (val * 0.5, val * 1.5)

            else:

                bounds['limit_'+key] = (val * 1.5, val * 0.5)

        self.boundary_parameter = bounds

    def pdf(self, x, mean, std, amplitude):

        pdf = (x - mean) / (np.sqrt(2) * std)
        pdf = - pdf**2
        pdf = np.exp(pdf)
        pdf = pdf * amplitude / (std * np.sqrt(2 * np.pi))

        return pdf

    def log_pdf(self, x, mean, std, amplitude):

        temp = np.log(amplitude) - np.log(std * np.sqrt(2 * np.pi))
        temp = temp - ((x - mean) / (std * np.sqrt(2)))**2

        return temp


if __name__ == '__main__':

    n_events, n_pixels, n_samples = 100, 1296, 50
    my_histo = Histogram1D(
                           data_shape=(n_pixels, ),
                           bin_edges=np.arange(-5, 5, 0.2),
                           )
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

    n_ac, n_dc, n_pixels, n_events, n_samples = 5, 5, 1, 1000, 50

    hist = Histogram1D(bin_edges=np.arange(-2000, 20000 + 1, 1),
                       data_shape=(n_ac, n_dc, n_pixels))

    for i in range(n_ac):
        for j in range(n_dc):
            for event_id in range(n_events):

                data = np.random.normal(size=(n_pixels, n_samples), loc=i,
                                        scale=j+1)

                hist.fill(data, indices=(i, j))

        hist.draw(index=(i, j, 0))

    hist.save('test.pk', compresslevel=0)
    hist.save('test_comp.pk')

    hist = Histogram1D.load('test_comp.pk')

    hist[0, 0, 0].draw()

    # bins = np.array([-10, -8, -6, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 6, 8, 10])
    bins = np.linspace(-100, 100, num=2000)
    hist = Histogram1D(bin_edges=bins)

    n_events = 1000
    n_samples = 1000
    for event_id in range(n_events):
        data = np.random.normal(size=(n_samples, ), loc=0, scale=5)

        hist.fill(data)

    fitter = MyHistogramFitter(hist, cost='MLE')
    fitter.fit(ncall=1000)
    fitter.compute_fit_errors()
    fitter.draw_fit()

    hist.draw()
    plt.show()
