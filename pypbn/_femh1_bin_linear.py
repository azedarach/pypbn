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


def _initialize_femh1_bin_linear_parameters_random(n_features, n_components,
                                                   random_state=None):
    rng = check_random_state(random_state)

    parameters = rng.uniform(size=(n_components, n_features))
    row_sums = np.sum(parameters, axis=1)
    parameters = parameters / (2 * row_sums[:, np.newaxis])

    return parameters


def _initialize_femh1_bin_linear_affiliations_random(n_samples, n_components,
                                                     random_state=None):
    rng = check_random_state(random_state)

    return left_stochastic_matrix((n_components, n_samples), random_state=rng)


def _initialize_femh1_bin_linear_parameters(n_features, n_components,
                                            init='random', random_state=None):
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_femh1_bin_linear_parameters_random(
            n_features, n_components, random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, INITIALIZATION_METHODS))


def _initialize_femh1_bin_linear_affiliations(n_samples, n_components,
                                              init='random',
                                              random_state=None):
    if init is None:
        init = 'random'

    if init == 'random':
        return _initialize_femh1_bin_linear_affiliations_random(
            n_samples, n_components, random_state=random_state)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, INITIALIZATION_METHODS))


def _initialize_femh1_bin_linear(X, n_components, init='random',
                                 random_state=None):
    if init is None:
        init = 'random'

    rng = check_random_state(random_state)

    n_features, n_samples = X.shape

    parameters = _initialize_femh1_bin_linear_parameters(
        n_features, n_components, init=init, random_state=rng)
    affiliations = _initialize_femh1_bin_linear_affiliations(
        n_samples, n_components, init=init, random_state=rng)

    return parameters, affiliations


def _fit_femh1_bin_linear_metropolis(Y, X, parameters, affiliations,
                                     epsilon_theta=0, epsilon_gamma=1e-6,
                                     sigma_theta=0.001, sigma_gamma=0.001,
                                     include_parameters=True,
                                     parameters_tolerance=1e-6,
                                     parameters_initialization=pypbn_ext.Uniform,
                                     chain_length=10000,
                                     max_parameters_iterations=1000,
                                     verbose=0, random_seed=0,
                                     print_frequency=10):
    if verbose:
        print('*** FEM-H1_BIN linear: n_components = {:d}'.format(
            affiliations.shape[0]))
        print('{:<12s} | {:<13s} | {:<13s} | {:<13s} | {:<13s} | {:<13s}'.format(
            'Iteration', 'Log-likelihood', 'Gamma acceptance',
            'Min. Theta acceptance', 'Max. Theta acceptance',
            'Average time'))
        print(70 * '-')

    stepper = pypbn_ext.FEMH1BinLinearMC(
        Y, X, parameters, affiliations,
        epsilon_theta=epsilon_theta, epsilon_gamma=epsilon_gamma,
        sigma_theta=sigma_theta, sigma_gamma=sigma_gamma,
        include_parameters=include_parameters,
        parameters_tolerance=parameters_tolerance,
        parameters_initialization=parameters_initialization,
        max_parameters_iterations=max_parameters_iterations,
        verbosity=(verbose > 1), random_seed=random_seed)

    average_time = 0
    for n_steps in range(chain_length):
        start_time = time.perf_counter()

        step_success = stepper.metropolis_step()

        end_time = time.perf_counter()

        step_time = end_time - start_time
        average_time = ((step_time + n_steps * average_time) /
                        (n_steps + 1))

        affiliations_acceptance_rate = stepper.get_affiliations_acceptance_rate()
        model_acceptance_rates = stepper.get_model_acceptance_rates()

        if verbose and (n_steps + 1) % print_frequency == 0:
            print('{:12d} | {: 12.6e} | {: 12.6e} | {:12.6e} | {: 12.6e} | {: 12.6e}'.format(
                n_steps + 1, log_like, affiliations_acceptance_rate,
                np.min(model_acceptance_rates),
                np.max(model_acceptance_rates), average_time))


class FEMH1BinLinear(object):
    def __init__(self, n_components, epsilon_theta=0, epsilon_gamma=1e-6,
                 init=None, sigma_theta=0.001, sigma_gamma=0.001,
                 include_parameters=True, parameters_tolerance=1e-6,
                 parameters_initialization=pypbn_ext.Uniform,
                 chain_length=1000, max_parameters_iterations=1000,
                 verbose=0, random_seed=0, random_state=None):
        self.n_components = n_components
        self.epsilon_theta = epsilon_theta
        self.epsilon_gamma = epsilon_gamma
        self.init = init
        self.sigma_theta = sigma_theta
        self.sigma_gamma = sigma_gamma
        self.include_parameters = include_parameters
        self.parameters_tolerance = parameters_tolerance
        self.parameters_initialization = parameters_initialization
        self.chain_length = chain_length
        self.max_parameters_iterations = max_parameters_iterations
        self.verbose = verbose
        self.random_seed = random_seed
        self.random_state = random_state

