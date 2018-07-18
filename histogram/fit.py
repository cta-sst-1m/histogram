from abc import abstractmethod
from iminuit import Minuit
from iminuit.util import describe
import numpy as np


class HistogramFitter:

    def __init__(self, histogram, pdf, log_pdf=None, print_level=0,
                 pedantic=False, cost='MLE', **kwargs):

        """

        :param histogram:
        :param pdf:
        :param log_pdf:
        :param print_level:
        :param pedantic:
        :param cost: Have a look at this article :
        https://ws680.nist.gov/publication/get_pdf.cfm?pub_id=914398
        :param kwargs:
        """

        if log_pdf is None:

            Warning('Trying to fit with the log of the PDF can lead'
                    'to numerical errors')

        self.histogram = histogram
        self.pdf = pdf
        self.log_pdf = log_pdf
        self.fitter = Minuit(self.cost_function, pedantic=pedantic,
                             print_level=print_level, **kwargs)
        self.cost = cost

        self.parameters_name = describe(pdf)[1:]

        self.initial_parameters = None
        self.boundary_parameter = None
        self.parameters = None
        self.parameters_errors = None

    @abstractmethod
    def initialize_fit(self):
        pass

    @abstractmethod
    def compute_fit_boundaries(self):
        pass

    def fit(self, **kwargs):

        self.initialize_fit()
        self.compute_fit_boundaries()

        options = self.fitter.fitarg
        options = {**options, **self.initial_parameters,
                   **self.boundary_parameter}

        self.fitter = Minuit(self.cost_function, **options,
                             forced_parameters=self.parameters_name)

        self.fitter.migrad(**kwargs)

        self.parameters = self.fitter.values
        self.parameters_errors = self.fitter.errors

    def compute_fit_errors(self, **kwargs):

        self.fitter.minos(**kwargs)
        self.parameters_errors = self.fitter.get_merrors()

    def cost_function(self, *params):

        if self.cost == 'MLE':

            return self._maximum_likelihood_estimator(*params)

        elif self.cost == 'NCHI2':

            return self._neymans_chi_square(*params)

        elif self.cost == 'PCHI2':

            return self._pearsons_chi_square(*params)

        else:

            raise ValueError('Invalid value for cost : {}'.format(self.cost))

    def _neymans_chi_square(self, *params):

        x = self.histogram.bin_centers
        bin_width = np.diff(self.histogram.bins)
        y = self.pdf(x, *params)
        count = self.histogram.data
        count = count / bin_width

        cost = (y - count)**2
        cost = cost / np.maximum(count, 1)
        cost = np.sum(cost, axis=-1)

        return cost

    def _pearsons_chi_square(self, *params):

        x = self.histogram.bin_centers
        bin_width = np.diff(self.histogram.bins)
        y = self.pdf(x, *params)
        count = self.histogram.data
        count = count / bin_width

        mask = y > 0
        count = count[:, mask]
        y = y[mask]

        cost = (y - count)**2 / y
        cost = np.sum(cost, axis=-1)

        return cost

    def _maximum_likelihood_estimator(self, *params):

        x = self.histogram.bin_centers
        bin_width = np.diff(self.histogram.bins)
        y = self.pdf(x, *params)

        count = self.histogram.data
        count = count / bin_width

        if self.log_pdf is None:

            log_y = np.log(y)

        else:

            log_y = self.log_pdf(x, *params)

        cost = y - count * log_y

        cost = np.sum(cost, axis=-1)

        return 2 * cost

    def draw(self, index, axes=None, **kwargs):

        axes = self.histogram.draw(index=index, axis=axes, color='k', **kwargs)
        bin_width = np.diff(self.histogram.bins)

        mask = self.histogram.data[index] > 0

        x_fit = self.histogram.bin_centers[mask]
        y_fit = self.pdf(x_fit, **self.parameters) * bin_width[mask]

        label_fit = 'Fit :\n'

        for key, val in self.parameters.items():

            label_fit += '{} : {:.2f} $\pm$ {:.3f}\n'.\
                format(key, val, self.parameters_errors[key])

        axes.plot(x_fit, y_fit, color='r', label=label_fit)
        axes.legend(loc='best')

        return axes


class MyHistogramFitter(HistogramFitter):

    def initialize_fit(self):

        x = self.histogram.bin_centers
        y = self.histogram.data[0]

        mean = np.average(x, weights=y)
        std = np.average((x - mean)**2, weights=y)
        std = np.sqrt(std)
        amplitude = np.sum(y, dtype=np.float)

        self.initial_parameters = {'mean':mean, 'std':std,
                                   'amplitude': amplitude}

    def compute_fit_boundaries(self):

        bounds = {}

        for key, val in self.initial_parameters.items():

            if val > 0:
                bounds['limit_'+key] = (val * 0.5, val * 1.5)

            else:

                bounds['limit_'+key] = (val * 1.5, val * 0.5)

        self.boundary_parameter = bounds


if __name__ == '__main__':

    from histogram import Histogram1D
    import matplotlib.pyplot as plt

    # bins = np.array([-10, -8, -6, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 6, 8, 10])
    bins = np.linspace(-100, 100 + 1, num=1000)
    hist = Histogram1D(bin_edges=bins,
                       data_shape=(1, ))

    def gaussian(x, mean, std, amplitude):

        temp = log_gaussian(x, mean, std, amplitude)
        temp = np.exp(temp)

        return temp

    def log_gaussian(x, mean, std, amplitude):

        temp = np.log(amplitude) - np.log(std * np.sqrt(2 * np.pi))
        temp = temp - ((x - mean) / (std * np.sqrt(2)))**2

        return temp

    n_events = 1000
    n_samples = 1000
    for event_id in range(n_events):
        data = np.random.normal(size=(n_samples, ), loc=0, scale=5)

        hist.fill(data, indices=(0, ))

    fitter = MyHistogramFitter(hist, pdf=gaussian, log_pdf=log_gaussian,
                               cost='MLE')
    fitter.fit(ncall=1000)
    # fitter.compute_fit_errors()
    fitter.draw(index=(0, ))

    hist.draw(index=(0, ))
    plt.show()