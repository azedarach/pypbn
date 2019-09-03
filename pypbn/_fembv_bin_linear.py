from __future__ import print_function

import numbers
import numpy as np
import time
import warnings

from sklearn.utils import check_array, check_random_state

from . import pypbn_ext
from ._random_matrix import left_stochastic_matrix


INTEGER_TYPES = (numbers.Integral, np.integer)

INITIALIZATION_METHODS = (None, 'random')


def _check_unit_axis_sums(A, whom, axis=0):
    axis_sums = np.sum(A, axis=axis)
    if not np.all(np.isclose(axis_sums, 1)):
        raise ValueError(
            'Array with incorrect axis sums passed to %s. '
            'Expected sums along axis %d to be 1.'
            % (whom, axis))


def _check_array_shape(A, shape, whom):
    if np.shape(A) != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, np.shape(A)))


def _check_init_affiliations(affiliations, shape, whom):
    affiliations = check_array(affiliations)
    _check_array_shape(affiliations, shape, whom)
    _check_unit_axis_sums(affiliations, whom, axis=0)


def _check_init_parameters(parameters, shape, whom):
    parameters = check_array(parameters)
    _check_array_shape(parameters, shape, whom)


def _initialize_fembv_bin_linear_parameters_random(n_features, n_components,
                                                   random_state=None):
    rng = check_random_state(random_state)

    parameters = rng.random(size=(n_components, n_features))
    row_sums = np.sum(parameters, axis=1)
    parameters = parameters / (2 * row_sums[:, np.newaxis])

    return parameters


def _initialize_fembv_bin_linear_affiliations_random(n_samples, n_components,
                                                     random_state=None):
    rng = check_random_state(random_state)

    return left_stochastic_matrix((n_components, n_samples), random_state=rng)


def _initialize_fembv_bin_linear_parameters(n_features, n_components,
                                            init='random', random_state=None):
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_fembv_bin_linear_parameters_random(
            n_features, n_components, random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, INITIALIZATION_METHODS))


def _initialize_fembv_bin_linear_affiliations(n_samples, n_components,
                                              init='random',
                                              random_state=None):
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_fembv_bin_linear_affiliations_random(
            n_samples, n_components, random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, INITIALIZATION_METHODS))


def _initialize_fembv_bin_linear(X, n_components, init='random',
                                 random_state=None):
    if init is None:
        init = 'random'

    rng = check_random_state(random_state)

    n_features, n_samples = X.shape

    parameters = _initialize_fembv_bin_linear_parameters(
        n_features, n_components, init=init, random_state=rng)
    affiliations = _initialize_fembv_bin_linear_affiliations(
        n_samples, n_components, init=init, random_state=rng)

    return parameters, affiliations


def _iterate_fembv_bin_linear(Y, X, parameters, affiliations,
                              epsilon=0, max_tv_norm=None,
                              update_parameters=True,
                              update_affiliations=True,
                              tolerance=1e-6,
                              parameters_tolerance=1e-6,
                              max_iterations=1000,
                              verbose=0,
                              require_monotonic_cost_decrease=True):
    if verbose:
        print('*** FEM-BV-BIN linear: n_components = {:d}'.format(
            affiliations.shape[0]))
        print('{:<12s} | {:<13s} | {:<13s} | {:<12s}'.format(
            'Iteration', 'Cost', 'Cost delta', 'Time'))
        print(60 * '-')

    solver = pybnd_ext.FEMBVBinLinear(Y, X, parameters, affiliations)
    solver.epsilon = epsilon
    if max_tv_norm is None:
        solver.max_tv_norm = -1
    else:
        solver.max_tv_norm = max_tv_norm

    old_cost = solver.cost()
    new_cost = old_cost

    parameters_success = True
    affiliations_success = True

    for n_iter in range(max_iterations):
        start_time = time.perf_counter()

        old_cost = new_cost

        if update_parameters:
            parameters_success = solver.update_parameters()
            new_cost = solver.cost()
            if (new_cost > old_cost) and require_monotonic_cost_decrease:
                raise RuntimeError(
                    'fit cost increased after parameters update')

        if update_affiliations:
            affiliations_success = solver.update_affiliations()
            new_cost = solver.cost()
            if (new_cost > old_cost) and require_monotonic_cost_decrease:
                raise RuntimeError(
                    'fit cost increased after affiliations update')

        cost_delta = new_cost - old_cost

        end_time = time.perf_counter()

        if verbose:
            print('{:12d} | {: 12.6e} | {: 12.6e} | {: 12.6e}'.format(
                n_iter + 1, new_cost, cost_delta, end_time - start_time))

        update_success = parameters_success and affiliations_success
        if abs(cost_delta) < tolerance and update_success:
            if verbose:
                print('*** Converged at iteration %d ***' % (n_iter + 1))
            break

    return solver.get_parameters(), solver.get_affiliations(), n_iter


def fembv_bin_linear_fit(outcome, predictors, parameters=None,
                         affiliations=None,
                         n_components=None, epsilon=0, max_tv_norm=None,
                         update_parameters=True, update_affiliations=True,
                         init=None, tolerance=1e-6, max_iterations=1000,
                         verbose=0, random_state=None):
    Y = outcome

    if predictors.ndim == 1:
        n_samples = predictors.shape[0]
        n_features = 1
        X = np.reshape(predictors, (n_features, n_samples))
    else:
        n_features, n_samples = predictors.shape
        X = predictors

    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError('Number of components must be a positive integer;'
                         ' got (n_components=%r)' % n_components)
    if not isinstance(max_iterations, INTEGER_TYPES) or max_iterations <= 0:
        raise ValueError('Maximum number of iterations must be a positive '
                         'integer; got (max_iterations=%r)' % max_iterations)
    if not isinstance(tolerance, numbers.Number) or tolerance < 0:
        raise ValueError('Tolerance for stopping criteria must be '
                         'positive; got (tolerance=%r)' % tolerance)

    if init == 'custom' and update_parameters and update_affiliations:
        _check_init_affiliations(affiliations, (n_components, n_samples),
                                 'fembv_bin_linear_fit (input affiliations)')
        _check_init_parameters(parameters, (n_components, n_features),
                               'fembv_bin_linear_fit (input parameters)')
    elif not update_parameters and update_affiliations:
        _check_init_parameters(parameters, (n_components, n_features),
                               'fembv_bin_linear_fit (input parameters)')
        affiliations = _initialize_fembv_bin_linear_affiliations(
            n_samples, n_components, init=init, random_state=random_state)
    elif update_parameters and not update_affiliations:
        _check_init_affiliations(affiliations, (n_components, n_samples),
                                 'fembv_bin_linear_fit (input affiliations)')
        parameters = _initialize_fembv_bin_linear_parameters(
            n_features, n_components, init=init, random_state=random_state)
    else:
        parameters, affiliations = _initialize_fembv_bin_linear(
            X, n_components, init=init, random_state=random_state)

    parameters, weights, n_iter, update_success = _iterate_fembv_bin_linear(
        Y, X, parameters, affiliations,
        epsilon=epsilon, max_tv_norm=max_tv_norm,
        update_parameters=update_parameters,
        update_affiliations=affiliations,
        tolerance=tolerance,
        parameters_tolerance=parameters_tolerance,
        max_iterations=max_iterations,
        verbose=verbose)

    if n_iter == max_iterations and tolerance > 0:
        warnings.warn('Maximum number of iterations %d reached.' %
                      max_iterations, UserWarning)

    if not update_success:
        warnings.warn('Update of model parameters failed.', UserWarning)

    return parameters, affiliations, n_iter


class FEMBVBinLinear(object):
    def __init__(self, n_components, epsilon=0, max_tv_norm=None,
                 init=None, tolerance=1e-6, max_iterations=1000,
                 verbose=0, random_state=None):
        self.n_components = n_components
        self.epsilon = epsilon
        self.init = init
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.random_state = random_state

    def fit_transform(self, outcomes, predictors, parameters=None,
                      affiliations=None):
        parameters_, affiliations_, n_iter_ = fembv_bin_linear_fit(
            outcomes, predictors, parameters=parameters,
            affiliations=affiliations,
            n_components=self.n_components,
            epsilon=self.epsilon, max_tv_norm=self.max_tv_norm,
            update_parameters=True, update_affiliations=True,
            init=self.init, tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            verbose=self.verbose, random_state=self.random_state)

        self.n_components_ = parameters.shape[0]
        self.parameters_ = parameters_
        self.n_iter_ = n_iter_

        return affiliations_

    def fit(self, outcomes, predictors, **kwargs):
        self.fit_transform(outcomes, predictors, **kwargs)
        return self

    def transform(self, outcomes, predictors):
        check_is_fitted(self, 'n_components')

        _, affiliations, _ = fembv_bin_linear_fit(
            outcomes, predictors, parameters=self.parameters_,
            n_components=self.n_components_,
            epsilon=self.epsilon, max_tv_norm=self.max_tv_norm,
            update_parameters=False, update_affiliations=True,
            init=self.init, tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            verbose=self.verbose, random_state=self.random_state)
