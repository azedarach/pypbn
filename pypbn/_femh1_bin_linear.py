from __future__ import division, print_function

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
                                     chain_length=10000, burn_in_fraction=0.5,
                                     max_parameters_iterations=1000,
                                     verbose=0, random_seed=0,
                                     print_frequency=10,
                                     observer=None):
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

    update_success = True
    average_time = 0
    average_parameters = np.zeros(parameters.shape)
    average_affiliations = np.zeros(affiliations.shape)
    max_log_likelihood = None
    for n_steps in range(chain_length):
        start_time = time.perf_counter()

        step_success = stepper.metropolis_step()
        update_success = update_success and step_success

        end_time = time.perf_counter()

        step_time = end_time - start_time

        average_time = ((step_time + n_steps * average_time) /
                        (n_steps + 1))

        parameters = stepper.get_parameters()
        affiliations = stepper.get_affiliations()

        log_like = stepper.get_log_likelihood()
        affiliations_acceptance_rate = stepper.get_affiliations_acceptance_rate()
        model_acceptance_rates = stepper.get_model_acceptance_rates()

        if n_steps / chain_length >= burn_in_fraction:
            average_parameters = ((parameters + n_steps * average_parameters) /
                                  (n_steps + 1))
            average_affiliations = ((affiliations + n_steps * average_affiliations) /
                                    (n_steps + 1))
            if max_log_likelihood is None or log_like > max_log_likelihood:
                max_log_likelihood = log_like

        if observer is not None:
            observer(parameters=parameters, affiliations=affiliations,
                     log_likelihood=log_like,
                     affiliations_acceptance_rate=affiliations_acceptance_rate,
                     model_acceptance_rates=model_acceptance_rates)

        if verbose and (n_steps + 1) % print_frequency == 0:
            print('{:12d} | {: 12.6e} | {: 12.6e} | {:12.6e} | {: 12.6e} | {: 12.6e}'.format(
                n_steps + 1, log_like, affiliations_acceptance_rate,
                np.min(model_acceptance_rates),
                np.max(model_acceptance_rates), average_time))

    return (average_parameters, average_affiliations, max_log_likelihood,
            update_success)


def femh1_bin_linear_fit_mc(outcome, predictors, parameters=None,
                            affiliations=None,
                            n_components=None,
                            epsilon_theta=0, epsilon_gamma=1e-6,
                            sigma_theta=0.001, sigma_gamma=0.001,
                            include_parameters=True,
                            init=None,
                            parameters_tolerance=1e-6,
                            parameters_initialization=pypbn_ext.Uniform,
                            chain_length=10000, burn_in_fraction=0.5,
                            max_parameters_iterations=1000,
                            verbose=0, random_seed=0, random_state=None,
                            print_frequency=10, observer=None):
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
    if not isinstance(chain_length, INTEGER_TYPES) or chain_length <= 0:
        raise ValueErrror('Chain length must be a positive integer;'
                          ' got (chain_length=%r)' % chain_length)

    if init == 'custom':
        _check_init_affiliations(affiliations, (n_components, n_samples),
                                 'femh1_bin_linear_fit_mc (input affiliations)')
        _check_init_parameters(parameters, (n_components, n_features),
                               'femh1_bin_linear_fit_mc (input parameters)')
    else:
        parameters, affiliations = _initialize_femh1_bin_linear(
            X, n_components, init=init, random_state=random_state)

    result = _fit_femh1_bin_linear_metropolis(
        Y, X, parameters, affiliations,
        epsilon_theta=epsilon_theta, epsilon_gamma=epsilon_gamma,
        sigma_theta=sigma_theta, sigma_gamma=sigma_gamma,
        include_parameters=include_parameters,
        parameters_tolerance=parameters_tolerance,
        parameters_initialization=parameters_initialization,
        chain_length=chain_length, burn_in_fraction=burn_in_fraction,
        max_parameters_iterations=max_parameters_iterations,
        verbose=verbose, random_seed=random_seed,
        print_frequency=print_frequency,
        observer=observer)

    update_success = result[-1]
    if not update_success:
        warnings.warn('Update of model parameters failed.', UserWarning)

    return result[:-1]


class FEMH1BinLinearMC(object):
    def __init__(self, n_components, epsilon_theta=0, epsilon_gamma=1e-6,
                 init=None, sigma_theta=0.001, sigma_gamma=0.001,
                 include_parameters=True, parameters_tolerance=1e-6,
                 parameters_initialization=pypbn_ext.Uniform,
                 chain_length=1000, burn_in_fraction=0.5,
                 max_parameters_iterations=1000,
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
        self.burn_in_fraction = burn_in_fraction
        self.max_parameters_iterations = max_parameters_iterations
        self.verbose = verbose
        self.random_seed = random_seed
        self.random_state = random_state

    def fit_transform(self, outcomes, predictors, parameters=None,
                      affiliations=None):
        parameters_, affiliations_, log_like_ = femh1_bin_linear_fit_mc(
            outcomes, predictors, parameters=parameters,
            affiliations=affiliations,
            n_components=self.n_components,
            epsilon_theta=self.epsilon_theta,
            epsilon_gamma=self.epsilon_gamma,
            sigma_theta=self.sigma_theta,
            sigma_gamma=self.sigma_gamma,
            include_parameters=self.include_parameters,
            init=self.init,
            parameters_tolerance=self.parameters_tolerance,
            parameters_initialization=self.parameters_initialization,
            chain_length=self.chain_length,
            burn_in_fraction=self.burn_in_fraction,
            max_parameters_iterations=self.max_parameters_iterations,
            verbose=self.verbose, random_seed=self.random_seed,
            random_state=self.random_state)

        self.n_components_ = parameters_.shape[0]
        self.parameters_ = parameters_
        self.log_likelihood_ = log_like_

        return affiliations_

    def fit(self, outcomes, predictors, **kwargs):
        self.fit_transform(outcomes, predictors, **kwargs)
        return self

