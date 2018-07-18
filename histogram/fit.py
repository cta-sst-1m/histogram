from abc import abstractmethod
import warnings

from iminuit import Minuit
from iminuit.util import describe
import numpy as np


class HistogramFitter:

    def __init__(self, histogram, parameters_plot_name={}, print_level=0,
                 pedantic=False, cost='MLE', **kwargs):

        """
        :param histogram:
        :param parameters_plot_name:
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

        out = self.compute_data_bounds()
        self.bin_centers = out[0]
        self.count = out[1]
        self.bin_width = out[2]

        self.fitter = None
        self.cost = cost
        self.ndf = np.nan

        self.initial_parameters = None
        self.boundary_parameter = None
        self.parameters = None
        self.errors = None
        self.minos_errors = {}
        self.parameters_plot_name = parameters_plot_name

    @abstractmethod
    def pdf(self, x, *params):
        pass

    def log_pdf(self, x, *params):

        warnings.warn('Trying to fit without the log of the PDF '
                      'can lead to numerical errors. Define a log_pdf()'
                      ' function in the class : {}'.
                      format(self.__class__.__name__),
                      RuntimeWarning)

        return np.log(self.pdf(x, *params))

    @abstractmethod
    def initialize_fit(self):
        pass

    @abstractmethod
    def compute_fit_boundaries(self):
        pass

    def compute_data_bounds(self):

        x = self.histogram.bin_centers
        y = self.histogram.data
        bin_width = np.diff(self.histogram.bins)

        mask = y > 0

        return x[mask], y[mask], bin_width[mask]

    def fit(self, **kwargs):

        self.initialize_fit()
        self.compute_fit_boundaries()

        options = {**self.iminuit_options,
                   **self.initial_parameters,
                   **self.boundary_parameter}

        self.fitter = Minuit(self.cost_function, **options)
        self.fitter.migrad(**kwargs)

        self.ndf = len(self.fitter.list_of_vary_param())
        self.ndf = len(self.count) - self.ndf

        self.parameters = self.fitter.values
        self.errors = self.fitter.errors

    def compute_fit_errors(self, **kwargs):

        try:
            self.fitter.minos(**kwargs)
            self.minos_errors = self.fitter.get_merrors()

        except Exception:

            pass

    def cost_function(self, *params):

        cost = self.cost

        count = self.count / self.bin_width
        y = self.pdf(self.bin_centers, *params)
        log_y = self.log_pdf(self.bin_centers, *params)

        if cost == 'MLE':

            return self._maximum_likelihood_estimator(y, log_y, count)

        elif cost == 'NCHI2':

            return self._neymans_chi_square(y, count)

        elif cost == 'PCHI2':

            return self._pearsons_chi_square(y, count)

        elif cost == 'MCHI2':

            return self._mighells_chi_square(y, count)

        elif cost == 'GMLE':

            return self._gauss_maximum_likelihood_estimator(y, log_y, count)

        else:

            raise ValueError('Invalid value for cost : {}'.format(self.cost))

    def _neymans_chi_square(self, y, count):

        cost = (y - count)**2
        cost = cost / np.maximum(count, 1)
        cost = np.sum(cost, axis=-1)

        return cost

    def _pearsons_chi_square(self, y, count):

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

    def fit_test(self):

        y = self.pdf(self.bin_centers, **self.parameters)
        y = y * self.bin_width
        chi2 = self._pearsons_chi_square(y, self.count)

        return chi2 / self.ndf

    def draw(self, index=(), axes=None, **kwargs):

        axes = self.histogram.draw(index=index, axis=axes, color='k', **kwargs)
        bin_width = np.diff(self.histogram.bins)

        mask = self.histogram.data[index] > 0

        x_fit = self.histogram.bin_centers[mask]
        y_fit = self.pdf(x_fit, **self.parameters) * bin_width[mask]

        label_fit = r'Fit : $\frac{\chi^2}{ndf}$' + ' : {:.2f}\n'.format(
            self.fit_test())
        line = '{} : {:.2f} $\pm$ {:.3f}\n'
        line_minos = '{name} : ' \
                     '${{{val:.2f}}}^{{+{upper:.3f}}}_{{{lower:.3f}}}$\n'

        for key, val in self.parameters.items():

            if key in self.parameters_plot_name.keys():

                name = self.parameters_plot_name[key]

            else:

                name = key

            if key in self.minos_errors.keys():

                lower = self.minos_errors[key]['lower']
                upper = self.minos_errors[key]['upper']
                label_fit += line_minos.format(name=name, val=val, upper=upper,
                                               lower=lower)

            else:

                label_fit += line.format(name, val, self.errors[key])

        axes.plot(x_fit, y_fit, color='r', label=label_fit)
        axes.legend(loc='best')

        return axes