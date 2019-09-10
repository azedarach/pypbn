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


def _fit_femh1_bin_linear_hmc(Y, X, parameters, affiliations,
                              epsilon_theta=0, epsilon_gamma=1e-6,
                              n_leapfrog_steps=10, leapfrog_step_size=0.001,
                              chain_length=10000,
                              verbose=0, random_seed=0, print_frequency=10):
    n_components = affiliations.shape[0]

    if verbose:
        print('*** FEM-H1-BIN linear HMC: n_components = {:d}'.format(
            n_components))
        print('{:<12s} | {:<13s} | {:<13s} | {:<13s}'.format(
            'Iteration', 'Log-likelihood', 'Acceptance',
            'Average time'))
        print(70 * '-')

    stepper = pypbn_ext.FEMH1BinLinearHMC(
        Y, X, parameters, affiliations,
        epsilon_theta=epsilon_theta, epsilon_gamma=epsilon_gamma,
        n_leapfrog_steps=n_leapfrog_steps,
        leapfrog_step_size=leapfrog_step_size,
        verbosity=(verbose > 1), random_seed=random_seed)

    update_success = True
    average_time = 0
    parameters_chain = np.zeros((chain_length,) + parameters.shape)
    affiliations_chain = np.zeros((chain_length,) + affiliations.shape)
    log_like_chain = np.zeros((chain_length,), dtype='f8')
    log_posterior_chain = np.zeros((chain_length,), dtype='f8')
    acceptance_rate_chain = np.zeros((chain_length,), dtype='f8')

    for n_steps in range(chain_length):
        start_time = time.perf_counter()

        step_success = stepper.hmc_step()
        update_success = update_success and step_success

        end_time = time.perf_counter()

        step_time = end_time - start_time

        average_time = ((step_time + n_steps * average_time) /
                        (n_steps + 1))

        current_parameters = stepper.get_parameters()
        current_affiliations = stepper.get_affiliations()
        log_like = stepper.get_log_likelihood()
        log_posterior = stepper.get_log_posterior()
        acceptance_rate = stepper.get_acceptance_rate()

        parameters_chain[n_steps] = current_parameters
        affiliations_chain[n_steps] = current_affiliations
        log_like_chain[n_steps] = log_like
        log_posterior_chain[n_steps] = log_posterior
        acceptance_rate_chain[n_steps] = acceptance_rate

        if verbose and (n_steps + 1) % print_frequency == 0:
            print('{:12d} | {: 12.6e} | {: 12.6e} | {:12.6e}'.format(
                n_steps + 1, log_like, acceptance_rate,
                average_time))

    return (parameters_chain, affiliations_chain, log_like_chain,
            log_posterior_chain, acceptance_rate_chain, update_success)


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
    n_components = affiliations.shape[0]

    if verbose:
        print('*** FEM-H1-BIN linear Metropolis: n_components = {:d}'.format(
            n_components))
        print('{:<12s} | {:<13s} | {:<13s} | {:<13s} | {:<13s} | {:<13s}'.format(
            'Iteration', 'Log-likelihood', 'Gamma acceptance',
            'Min. Theta acceptance', 'Max. Theta acceptance',
            'Average time'))
        print(70 * '-')

    stepper = pypbn_ext.FEMH1BinLinearMH(
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
    affiliations_acceptance_rate = 0
    model_acceptance_rates = None
    parameters_chain = np.zeros((chain_length,) + parameters.shape)
    affiliations_chain = np.zeros((chain_length,) + affiliations.shape)
    log_like_chain = np.zeros((chain_length,), dtype='f8')
    log_posterior_chain = np.zeros((chain_length,), dtype='f8')
    affiliations_acceptance_rate_chain = np.zeros((chain_length,), dtype='f8')
    model_acceptance_rates_chain = np.zeros((chain_length, n_components,),
                                            dtype='f8')

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
        log_posterior = stepper.get_log_posterior()

        parameters_chain[n_steps] = current_parameters
        affiliations_chain[n_steps] = current_affiliations
        log_like_chain[n_steps] = log_like
        log_posterion_chain[n_steps] = log_posterior

        affiliations_acceptance_rate = stepper.get_affiliations_acceptance_rate()
        model_acceptance_rates = stepper.get_model_acceptance_rates()

        affiliations_acceptance_rate_chain[n_steps] = affiliations_acceptance_rate
        model_acceptance_rates_chain[n_steps] = model_acceptance_rates

        if verbose and (n_steps + 1) % print_frequency == 0:
            print('{:12d} | {: 12.6e} | {: 12.6e} | {:12.6e} | {: 12.6e} | {: 12.6e}'.format(
                n_steps + 1, log_like, affiliations_acceptance_rate,
                np.min(model_acceptance_rates),
                np.max(model_acceptance_rates), average_time))

    return (parameters_chain, affiliations_chain, log_like_chain,
            log_posterior_chain
            affiliations_acceptance_rate_chain,
            model_acceptance_rates_chain,
            update_success)


def femh1_bin_linear_fit_mc(outcome, predictors, parameters=None,
                            affiliations=None,
                            n_components=None,
                            epsilon_theta=0, epsilon_gamma=1e-6,
                            method='metropolis',
                            sigma_theta=0.001, sigma_gamma=0.001,
                            n_leapfrog_steps=10, leapfrog_step_size=0.001,
                            include_parameters=True,
                            init=None,
                            parameters_tolerance=1e-6,
                            parameters_initialization=pypbn_ext.Uniform,
                            chain_length=10000,
                            max_parameters_iterations=1000,
                            verbose=0, random_seed=0, random_state=None,
                            print_frequency=10):
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

    if method == 'metropolis':
        result = _fit_femh1_bin_linear_metropolis(
            Y, X, parameters, affiliations,
            epsilon_theta=epsilon_theta, epsilon_gamma=epsilon_gamma,
            sigma_theta=sigma_theta, sigma_gamma=sigma_gamma,
            include_parameters=include_parameters,
            parameters_tolerance=parameters_tolerance,
            parameters_initialization=parameters_initialization,
            chain_length=chain_length,
            max_parameters_iterations=max_parameters_iterations,
            verbose=verbose, random_seed=random_seed,
            print_frequency=print_frequency)
    elif method == 'hmc':
        result = _fit_femh1_bin_linear_hmc(
            Y, X, parameters, affiliations,
            epsilon_theta=epsilon_theta, epsilon_gamma=epsilon_gamma,
            n_leapfrog_steps=n_leapfrog_steps,
            leapfrog_step_size=leapfrog_step_size,
            chain_length=chain_length,
            verbose=verbose, random_seed=random_seed,
            print_frequency=print_frequency)
    else:
        raise ValueError("Invalid method parameter '%r'" % method)

    update_success = result[-1]
    if not update_success:
        warnings.warn('Update of model parameters failed.', UserWarning)

    return result[:-1]


class FEMH1BinLinearMC(object):
    def __init__(self, n_components, epsilon_theta=0, epsilon_gamma=1e-6,
                 init=None, method='metropolis',
                 sigma_theta=0.001, sigma_gamma=0.001,
                 n_leapfrog_steps=10, leapfrog_step_size=0.001,
                 include_parameters=True, parameters_tolerance=1e-6,
                 parameters_initialization=pypbn_ext.Uniform,
                 chain_length=1000,
                 max_parameters_iterations=1000,
                 verbose=0, random_seed=0, random_state=None):
        self.n_components = n_components
        self.epsilon_theta = epsilon_theta
        self.epsilon_gamma = epsilon_gamma
        self.init = init
        self.method = method
        self.sigma_theta = sigma_theta
        self.sigma_gamma = sigma_gamma
        self.n_leapfrog_steps = n_leapfrog_steps
        self.leapfrog_step_size = leapfrog_step_size
        self.include_parameters = include_parameters
        self.parameters_tolerance = parameters_tolerance
        self.parameters_initialization = parameters_initialization
        self.chain_length = chain_length
        self.max_parameters_iterations = max_parameters_iterations
        self.verbose = verbose
        self.random_seed = random_seed
        self.random_state = random_state

    def fit_transform(self, outcomes, predictors, parameters=None,
                      affiliations=None):
        result = femh1_bin_linear_fit_mc(
            outcomes, predictors, parameters=parameters,
            affiliations=affiliations,
            n_components=self.n_components,
            epsilon_theta=self.epsilon_theta,
            epsilon_gamma=self.epsilon_gamma,
            sigma_theta=self.sigma_theta,
            sigma_gamma=self.sigma_gamma,
            n_leapfrog_steps=self.n_leapfrog_steps,
            leapfrog_step_size=self.leapfrog_step_size,
            include_parameters=self.include_parameters,
            init=self.init,
            method=self.method,
            parameters_tolerance=self.parameters_tolerance,
            parameters_initialization=self.parameters_initialization,
            chain_length=self.chain_length,
            max_parameters_iterations=self.max_parameters_iterations,
            verbose=self.verbose, random_seed=self.random_seed,
            random_state=self.random_state)[:3]

        self.parameters_chain_ = result[0]
        self.n_components_ = self.parameters_chain_.shape[1]
        self.log_likelihood_chain_ = result[2]
        self.log_posterior_chain_ = result[3]

        if self.method == 'metropolis':
            self.affiliations_acceptance_rate_chain_ = result[4]
            self.model_acceptance_rates_chain_ = result[5]
        else:
            self.acceptance_rate_chain_ = result[4]

        return result[1]

    def fit(self, outcomes, predictors, **kwargs):
        self.fit_transform(outcomes, predictors, **kwargs)
        return self

