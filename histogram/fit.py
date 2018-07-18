from abc import abstractmethod
import warnings

from iminuit import Minuit
from iminuit.util import describe
import numpy as np


class HistogramFitter:

    def __init__(self, histogram, print_level=0,
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

        self.histogram = histogram
        self.parameters_name = describe(self.pdf)[1:]
        self.iminuit_options = {'pedantic': pedantic,
                                'print_level': print_level,
                                'forced_parameters': self.parameters_name,
                                **kwargs}

        self.fitter = None
        self.cost = cost

        self.initial_parameters = None
        self.boundary_parameter = None
        self.parameters = None
        self.parameters_errors = None

    @abstractmethod
    def pdf(self, x, *params):
        pass

    def log_pdf(self, x, *params):

        warnings.warn('Trying to fit without the log of the PDF '
                      'can lead to numerical errors. Define a log_pdf()'
                      'function in the class : {}'.
                      format(self.__class__.__name__),
                      RuntimeWarning)

        return np.log(self.pdf(x, *params))

    @abstractmethod
    def initialize_fit(self):
        pass

    @abstractmethod
    def compute_fit_boundaries(self):
        pass

    def fit(self, **kwargs):

        self.initialize_fit()
        self.compute_fit_boundaries()

        options = {**self.iminuit_options,
                   **self.initial_parameters,
                   **self.boundary_parameter}

        self.fitter = Minuit(self.cost_function, **options)
        self.fitter.migrad(**kwargs)

        self.parameters = self.fitter.values
        self.parameters_errors = self.fitter.errors

    def compute_fit_errors(self, **kwargs):

        self.fitter.minos(**kwargs)
        self.parameters_errors = self.fitter.get_merrors()

    def cost_function(self, *params):

        x = self.histogram.bin_centers
        bin_width = np.diff(self.histogram.bins)
        y = self.pdf(x, *params)
        log_y = self.log_pdf(x, *params)
        count = self.histogram.data
        count = count / bin_width

        if self.cost == 'MLE':

            return self._maximum_likelihood_estimator(y, log_y, count)

        elif self.cost == 'NCHI2':

            return self._neymans_chi_square(y, count)

        elif self.cost == 'PCHI2':

            return self._pearsons_chi_square(y, count)

        elif self.cost == 'MCHI2':

            return self._mighells_chi_square(y, count)

        elif self.cost == 'GMLE':

            return self._gauss_maximum_likelihood_estimator(y, log_y, count)

        else:

            raise ValueError('Invalid value for cost : {}'.format(self.cost))

    def _neymans_chi_square(self, y, count):

        cost = (y - count)**2
        cost = cost / np.maximum(count, 1)
        cost = np.sum(cost, axis=-1)

        return cost

    def _pearsons_chi_square(self, y, count):

        mask = y > 0
        count = count[:, mask]
        y = y[mask]

        cost = (y - count)**2 / y
        cost = np.sum(cost, axis=-1)

        return cost

    def _mighells_chi_square(self, y, count):

        cost = count + np.minimum(count, 1) + y
        cost = cost**2
        cost = cost / (count + 1)
        cost = np.sum(cost, axis=-1)

        return cost

    def _maximum_likelihood_estimator(self, y, log_y, count):

        cost = y - count * log_y
        cost = np.sum(cost, axis=-1)

        return 2 * cost

    def _gauss_maximum_likelihood_estimator(self, y, log_y, count):

        cost = self._pearsons_chi_square(y, count)
        cost = cost + np.sum(log_y, axis=-1)

        return cost

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
