from __future__ import division

import argparse
import datetime
import numpy as np
import pypbn
import shelve
import time

from pypbn import FEMH1BinLinearMC


DEFAULT_YEAR_FIELD = 'year'
DEFAULT_MONTH_FIELD = 'month'
DEFAULT_DAY_FIELD = 'day'
DEFAULT_DUMMY_DAY = 1


def read_data(csv_file, year_field=DEFAULT_YEAR_FIELD,
              month_field=DEFAULT_MONTH_FIELD,
              day_field=DEFAULT_DAY_FIELD,
              dummy_day=DEFAULT_DUMMY_DAY):
    data = np.genfromtxt(csv_file, delimiter=',', names=True)

    years = np.array(data[year_field], dtype='i8')
    months = np.array(data[month_field], dtype='i8')
    if day_field in data.dtype.names:
        is_daily_data = True
        days = np.array(data[day_field], dtype='i8')
    else:
        is_daily_data = False
        days = np.full(years.shape, dummy_day, dtype='i8')

    times = np.array([datetime.datetime(years[i], months[i], days[i])
                      for i in range(np.size(years))])

    fields = [f for f in data.dtype.names
              if f not in (year_field, month_field, day_field)]
    values = {f: np.array(data[f]) for f in fields}

    return times, values, is_daily_data


def to_categorical_values(data, zero_positive=False):
    categorical_data = {f: None for f in data}

    for f in data:
        signs_data = np.sign(data[f])

        if zero_positive:
            signs_data[signs_data == 0] = 1
        else:
            signs_data[signs_data == 0] = -1
        signs_data[signs_data < 0] = 0

        categorical_data[f] = signs_data

    return categorical_data


def generate_fit_inputs(times, categorical_values, lags=None,
                        double_indicators=False):
    if lags is None or not lags:
        max_lag = 0
        n_lags = 0
    else:
        max_lag = np.max(lags)
        n_lags = np.size(lags)

    lagged_times = times[max_lag:]

    fields = [f for f in categorical_values]
    lagged_outcomes = {f: categorical_values[f][max_lag:]
                       for f in categorical_values}

    n_samples = lagged_times.shape[0]
    n_features = len(lagged_outcomes)

    if lags is None or not lags:
        # intercept term only
        predictors = np.ones((1, n_samples), dtype='i8')
        predictor_names = ['unresolved']
    else:
        if double_indicators:
            n_indicators = 2 * n_features * n_lags
        else:
            n_indicators = n_features * n_lags

        if 0 in lags:
            n_predictors = n_indicators + 1
            predictor_names = ['unresolved']
            predictors = np.zeros((n_predictors, n_samples), dtype='i8')
            predictors[0, :] = 1
            index = 1
        else:
            n_predictors = n_indicators
            predictor_names = []
            predictors = np.zeros((n_predictors, n_samples), dtype='i8')
            index = 0

        for lag in lags:
            if lag == 0:
                continue
            for i, f in enumerate(fields):
                lagged_field = categorical_values[f][max_lag - lag:-lag]
                if double_indicators:
                    predictors[index, :][lagged_field == 1] = 1
                    index += 1
                    predictor_names.append('{}_positive_lag_{:d}'.format(f, lag))
                    predictors[index, :][lagged_field == 0] = 1
                    index += 1
                    predictor_names.append('{}_negative_lag_{:d}'.format(f, lag))
                else:
                    predictors[index, :][lagged_field == 1] = 1
                    index += 1
                    predictor_names.append('{}_positive_lag_{:d}'.format(f, lag))

    return lagged_times, lagged_outcomes, predictors, predictor_names


def calculate_within_chain_moments(parameters_batch, affiliations_batch):
    n_components = parameters_batch.shape[-2]
    n_features = parameters_batch.shape[-1]
    n_samples = affiliations_batch.shape[-1]

    parameters_within_chain_means = np.mean(parameters_batch, axis=1)
    parameters_within_chain_vars = np.var(parameters_batch, axis=1, ddof=1)

    affiliations_within_chain_means = np.mean(affiliations_batch, axis=1)
    affiliations_within_chain_vars = np.var(affiliations_batch, axis=1, ddof=1)

    # variances and covariances necessary for estimating degrees of freedom
    parameters_var_var = np.var(
        parameters_within_chain_vars, axis=0, ddof=1)
    affiliations_var_var = np.var(
        affiliations_within_chain_vars, axis=0, ddof=1)

    parameters_var_mean_sq_cov = np.empty((n_components, n_features))
    parameters_var_mean_cov = np.empty((n_components, n_features))

    for j in range(n_components):
        for k in range(n_features):
            parameters_var_mean_sq_cov[j, k] = np.cov(
                parameters_within_chain_vars[:, j, k],
                parameters_within_chain_means[:, j, k] **2,
                ddof=1)
            parameters_var_mean_cov[j, k] = np.cov(
                parameters_within_chain_vars[:, j, k],
                parameters_within_chain_means[:, j, k],
                ddof=1)

    affiliations_var_mean_sq_cov = np.empty((n_components, n_samples))
    affiliations_var_mean_cov = np.empty((n_components, n_samples))
    for j in range(n_components):
        for k in range(n_samples):
            affiliations_var_mean_sq_cov[j, k] = np.cov(
                affiliations_within_chain_vars[:, j, k],
                affiliations_within_chain_means[:, j, k] ** 2,
                ddof=1)
            affiliations_var_mean_cov[j, k] = np.cov(
                affiliations_within_chain_vars[:, j, k],
                affiliations_chain_means[:, j, k],
                ddof=1)

    return dict(parameters_within_chain_means=parameters_within_chain_means,
                parameters_within_chain_vars=parameters_within_chain_vars,
                affiliations_within_chain_means=affiliations_within_chain_means,
                affiliations_within_chain_vars=affiliations_within_chain_vars,
                parameters_var_var=parameters_var_var,
                parameters_var_mean_sq_cov=parameters_var_mean_sq_cov,
                parameters_var_mean_cov=parameters_var_mean_cov,
                affiliations_var_var=affiliations_var_var,
                affiliations_var_mean_sq_cov=affiliations_var_mean_sq_cov,
                affiliations_var_mean_cov=affiliations_var_mean_cov)


def calculate_diagnostics(parameters_chains, affiliations_chains,
                          burn_in_fraction=0.5, batch_size=None):
    n_chains = parameters_chains.shape[0]
    chain_length = parameters_chains.shape[1]
    n_components = parameters_chains.shape[2]
    n_features = parameters_chains.shape[3]
    n_samples = affiliations_chains.shape[-1]

    if batch_size is None:
        batch_size = 0.05 * burn_in_fraction * chain_length

    n_batches = int(np.ceil(burn_in_fraction * chain_length / batch_size))

    parameters_W = np.full(
        (n_batches, n_components, n_features), np.NaN)
    parameters_Vhat = np.full(
        (n_batches, n_components, n_features), np.NaN)
    parameters_Rhat = np.full(
        (n_batches, n_components, n_features), np.NaN)
    parameters_dof = np.full(
        (n_batches, n_components, n_features), np.NaN)

    affiliations_W = np.full(
        (n_batches, n_components, n_samples), np.NaN)
    affiliations_Vhat = np.full(
        (n_batches, n_components, n_samples), np.NaN)
    affiliations_Rhat = np.full(
        (n_batches, n_components, n_samples), np.NaN)
    affiliations_dof = np.full(
        (n_batches, n_components, n_samples), np.NaN)

    start_pos = int(np.floor(burn_in_fraction * chain_length))
    for i in range(n_batches):
        end_pos = min(chain_length, start_pos + (i + 1) * batch_size)

        parameters_batch = parameters_chains[:, start_pos:end_pos, :, :]
        affiliations_batch = affiliations_chains[:, start_pos:end_pos, :, :]

        current_batch_size = parameters_batch.shape[1]

        within_chain_moments = calculate_within_chain_moments(
            parameters_batch, affiliations_batch)

        parameters_within_chain_means = within_chain_moments[
            'parameters_within_chain_means']
        affiliations_within_chain_means = within_chain_moments[
            'affiliations_within_chain_means']

        parameters_W[i] = np.mean(
            within_chain_moments['parameters_within_chain_vars'], axis=0)
        affiliations_W[i] = np.mean(
            within_chain_moments['affiliations_within_chain_vars'], axis=0)

        parameters_combined_means = np.mean(parameters_batch, axis=[0, 1],
                                            keepdims=True)
        affiliations_combined_means = np.mean(affiliations_batch, axis=[0, 1],
                                              keepdims=True)

        parameters_mean_var = (
            np.sum(
                (parameters_within_chain_means -
                 parameters_combined_means[np.newaxis, :, :]) ** 2,
                axis=0) /
            (n_chains - 1))
        affiliations_mean_var = (
            np.sum(
                (affiliations_within_chain_means -
                 affiliations_combined_means[np.newaxis, :, :]) ** 2,
                axis=0) /
            (n_chains - 1))

        parameters_sigma_hat = (
            (current_batch_size - 1) * parameters_W[i] /
            current_batch_size + parameters_mean_var)
        affiliations_sigma_hat = (
            (current_batch_size - 1) * affiliations_W[i] /
            current_batch_size + affiliations_mean_var)

        parameters_Vhat[i] = (parameters_sigma_hat +
                              parameters_mean_var / n_chains)
        affiliations_Vhat[i] = (affiliations_sigma_hat +
                                affiliations_mean_var / n_chains)

        parameters_Rhat[i] = (parameters_Vhat[i] /
                              parameters_W[i])
        affiliations_Rhat[i] = (affiliations_Vhat[i] /
            affiliations_W[i])

        parameters_var_Vhat = (
            ((current_batch_size - 1.0) / current_batch_size) ** 2 *
            within_chain_moments['parameters_var_var'] / n_chains +
            2 * ((n_chains + 1.0) / n_chains) ** 2 *
            parameters_mean_var ** 2 / (n_chains - 1.0) +
            2 * ((n_chains + 1.0) * (current_batch_size - 1.0) /
                 (n_chains * current_batch_size ** 2)) *
            current_batch_size * (
                within_chain_moments['parameters_var_mean_sq_cov'] -
                2 * parameters_combined_means *
                within_chain_moments['parameters_var_mean_cov']) / n_chains)

        affiliations_var_Vhat = (
            ((current_batch_size - 1.0) / current_batch_size) ** 2 *
            within_chain_moments['affiliations_var_var'] / n_chains +
            2 * ((n_chains + 1.0) / n_chains) ** 2 *
            affiliations_mean_var ** 2 / (n_chains - 1.0) +
            2 * ((n_chains + 1.0) * (current_batch_size - 1.0) /
                 (n_chains * current_batch_size ** 2)) *
            current_batch_size * (
                within_chain_moments['affiliations_var_mean_sq_cov'] -
                2 * affiliations_combined_means *
                within_chain_moments['affiliations_var_mean_cov']) / n_chains)

        parameters_dof[i] = 2 * parameters_Vhat[i] / parameters_var_Vhat
        affiliations_dof[i] = 2 * affiliations_Vhat[i] / affiliations_var_Vhat

        if end_pos >= chain_length:
            break

    parameters_Rhatc = ((parameters_dof + 3) * parameters_Rhat /
                        (parameters_dof + 1))
    affiliations_Rhatc = ((affiliations_dof + 3) * affiliations_Rhat /
                          (affiliations_dof + 1))

    return dict(parameters_W=parameters_W,
                parameters_Vhat=parameters_Vhat,
                parameters_Rhat=parameters_Rhat,
                parameters_dof=parameters_dof,
                parameters_Rhatc=parameters_Rhatc,
                affiliations_W=affiliations_W,
                affiliations_Vhat=affiliations_Vhat,
                affiliations_Rhat=affiliations_Rhat,
                affiliations_dof=affiliations_dof,
                affiliations_Rhatc=affiliations_Rhatc)


def run_femh1_linear_mcmc(Y, X, n_components=None, n_chains=1,
                          epsilon_theta=0, epsilon_gamma=1e-6,
                          init=None, method='metropolis',
                          sigma_theta=0.001, sigma_gamma=0.001,
                          n_leapfrog_steps=10, leapfrog_step_size=0.001,
                          parameters_tolerance=1e-6,
                          parameters_initialization=None,
                          chain_length=10000, burn_in_fraction=0.5,
                          max_parameters_iterations=1000,
                          verbose=0, random_seed=0, random_state=None):
    if X.ndim == 1:
        n_features = 1
        n_samples = np.size(X)
    else:
        n_features, n_samples = X.shape

    affiliations_chains = np.empty(
        (n_chains, chain_length, n_components, n_samples))
    parameters_chains = np.empty(
        (n_chains, chain_length, n_components, n_features))
    log_likelihood_chains = np.empty(
        (n_chains, chain_length))
    runtimes = np.empty((n_chains,))

    success = False
    for i in range(n_chains):
        start_time = time.perf_counter()

        model = pypbn.FEMH1BinLinearMC(n_components=n_components,
                                       epsilon_theta=epsilon_theta,
                                       epsilon_gamma=epsilon_gamma,
                                       init=init, method=method,
                                       sigma_theta=sigma_theta,
                                       sigma_gamma=sigma_gamma,
                                       n_leapfrog_steps=n_leapfrog_steps,
                                       leapfrog_step_size=leapfrog_step_size,
                                       parameters_tolerance=parameters_tolerance,
                                       parameters_initialization=parameters_initialization,
                                       chain_length=chain_length,
                                       max_parameters_iterations=max_parameters_iterations,
                                       verbose=verbose, random_seed=random_seed,
                                       random_state=random_state)

        affiliations_chain = model.fit_transform(Y, X)
        parameters_chain = model.parameters_chain_
        log_likelihood_chain = model.log_likelihood_chain_

        end_time = time.perf_counter()

        runtime = end_time - start_time

        affiliations_chains[i] = affiliations_chain
        parameters_chains[i] = parameters_chain
        log_likelihood_chains[i] = log_likelihood_chain
        runtimes[i] = runtime

        success = True

    if not success:
        raise RuntimeError('failed to fit FEM-H1 model')

    diagnostics = calculate_diagnostics(parameters_chains, affiliations_chains)

    return dict(affiliations=affiliations_chains,
                parameters=parameters_chains,
                log_likelihood=log_likelihood_chains,
                runtime=runtimes, diagnostics=diagnostics)


def create_model_dict(outcome, epsilon_gamma, epsilon_theta,
                      times, predictor_names, fit_result, attrs=None):
    n_components = fit_result['affiliations'].shape[-2]
    model = {'outcome': outcome,
             'epsilon_gamma': epsilon_gamma,
             'epsilon_theta': epsilon_theta,
             'n_components': n_components,
             'affiliations': fit_result['affiliations'].copy(),
             'time': times,
             'log_likelihood': fit_result['log_likelihood'],
             'runtime': fit_result['runtime']
             }

    components = []
    for i in range(n_components):
        component_parameters = fit_result['parameters'][:, :, i, :]
        component = {p: component_parameters[:, :, pi]
                     for pi, p in enumerate(predictor_names)}
        components.append(component)

    model['components'] = components

    model['diagnostics'] = fit_result['diagnostics']

    if attrs is not None:
        for attr in attrs:
            model[attr] = attrs[attr]

    return model


def write_predictors(output_file, times, predictors, is_daily_data=False,
                     year_field=DEFAULT_YEAR_FIELD,
                     month_field=DEFAULT_MONTH_FIELD,
                     day_field=DEFAULT_DAY_FIELD,
                     predictor_names=None):
    if predictors.ndim == 1:
        n_samples = np.size(predictors)
        n_predictors = 1
    else:
        n_predictors, n_samples = predictors.shape

    if predictor_names is None:
        predictor_names = ['x{:d}'.format(i) for i in range(n_predictors)]

    if is_daily_data:
        n_fields = 3 + n_predictors
        header = ','.join([year_field, month_field, day_field] +
                          predictor_names)
        fmt = ','.join(['%d', '%d', '%d'] + ['%d'] * n_predictors)
    else:
        n_fields = 2 + n_predictors
        header = ','.join([year_field, month_field] +
                          predictor_names)
        fmt = ','.join(['%d', '%d'] + ['%d'] * n_predictors)

    output_data = np.empty((n_samples, n_fields))

    output_data[:, 0] = np.array([dt.year for dt in times], dtype='i8')
    output_data[:, 1] = np.array([dt.month for dt in times], dtype='i8')

    if is_daily_data:
        output_data[:, 2] = np.array([dt.day for dt in times], dtype='i8')
        output_data[:, 3:] = predictors.T
    else:
        output_data[:, 2:] = predictors.T

    np.savetxt(output_file, output_data, header=header, fmt=fmt)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Run FEM-H1-binary fit on categorical data')

    parser.add_argument('input_csv_file',
                        help='file containing input data')
    parser.add_argument('--n-components', dest='n_components',
                        type=int, default=1,
                        help='number of clusters')
    parser.add_argument('--epsilon-theta', dest='epsilon_theta',
                        type=float, action='append',
                        help='regularization parameter')
    parser.add_argument('--epsilon-gamma', dest='epsilon_gamma',
                        type=float, action='append',
                        help='affiliations regularization parameter')
    parser.add_argument('--sigma-theta', dest='sigma_theta',
                        type=float, default=0.001,
                        help='parameters step size')
    parser.add_argument('--sigma-gamma', dest='sigma_gamma',
                        type=float, default=0.001,
                        help='affiliations step size')
    parser.add_argument('--n-leap-frog-steps', dest='n_leapfrog_steps',
                        type=int, default=10,
                        help='number of leapfrog steps')
    parser.add_argument('--leapfrog-step-size', dest='leapfrog_step_size',
                        type=float, default=0.001,
                        help='leapfrog step size')
    parser.add_argument('--lag', dest='lag', type=int, action='append',
                        help='lag to be included in model')
    parser.add_argument('--n-init', dest='n_init', type=int,
                        default=1, help='number of chains')
    parser.add_argument('--init', dest='init',
                        choices=['random'], default='random',
                        help='initialization method')
    parser.add_argument('--method', dest='method',
                        choices=['metropolis', 'hmc'], default='metropolis',
                        help='MCMC algorithm')
    parser.add_argument('--parameters-tolerance', dest='parameters_tolerance',
                        type=float, default=1e-6,
                        help='parameters optimization tolerance')
    parser.add_argument('--parameters-initialization',
                        dest='parameters_initialization',
                        choices=[pypbn.Uniform, pypbn.Random, pypbn.Current],
                        default=pypbn.Uniform,
                        help='parameters optimization initial guess method')
    parser.add_argument('--chain-length', dest='chain_length',
                        type=int, default=10000,
                        help='chain length')
    parser.add_argument('--max-parameters-iterations',
                        dest='max_parameters_iterations',
                        type=int, default=10000,
                        help='maximum number of parameter optimizer iterations')
    parser.add_argument('--outcome', dest='outcome', action='append',
                        help='name of outcome to fit')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='produce verbose output')
    parser.add_argument('--random-seed', dest='random_seed', type=int,
                        default=0,
                        help='random seed used by parameters optimizer')
    parser.add_argument('--random-state', dest='random_state', type=int,
                        default=None,
                        help='random state used for initial guess')
    parser.add_argument('--output-file', dest='output_file',
                        default='', help='output file to write fits to')
    parser.add_argument('--predictors-file', dest='predictors_file',
                        default='', help='output file to write predictors to')
    parser.add_argument('--double-indicators', dest='double_indicators',
                        action='store_true',
                        help='include redundant predictors')

    return parser.parse_args()


def main():
    args = parse_cmd_line_args()

    times, values, is_daily_data = read_data(args.input_csv_file)

    categorical_values = to_categorical_values(values)

    if not args.lag:
        lags = [0]
    else:
        lags = []
        for lag in args.lag:
            if lag < 0:
                raise ValueError('received invalid lag value %r' % lag)
            if lag in lags:
                continue
            else:
                lags.append(lag)
        lags = sorted(lags)

    times, outcomes, predictors, predictor_names = generate_fit_inputs(
        times, categorical_values, lags=lags,
        double_indicators=args.double_indicators)

    if args.predictors_file:
        write_predictors(args.predictors_file, times, predictors,
                         is_daily_data=is_daily_data,
                         predictor_names=predictor_names)

    if not args.epsilon_theta:
        epsilon_thetas = [0]
    else:
        epsilon_thetas = args.epsilon_theta

    if not args.epsilon_gamma:
        epsilon_gammas = [1e-6]
    else:
        epsilon_gammas = args.epsilon_gamma

    if args.outcome is None or not args.outcome:
        outcome_names = [f for f in outcomes]
    else:
        for o in args.outcome:
            if o not in outcomes:
                raise ValueError('invalid outcome %s' % o)
        outcome_names = args.outcome

    if args.parameters_initialization == pypbn.Uniform:
        parameters_init_string = 'Uniform'
    elif args.parameters_initialization == pypbn.Random:
        parameters_init_string = 'Random'
    elif args.parameters_initialization == pypbn.Current:
        parameters_init_string = 'Current'
    else:
        raise ValueError('invalid parameters initialization %r' %
                         args.parameters_initialization)

    attrs = dict(n_components=args.n_components,
                 n_init=args.n_init,
                 sigma_theta=args.sigma_theta,
                 sigma_gamma=args.sigma_gamma,
                 n_leapfrog_steps=args.n_leapfrog_steps,
                 leapfrog_step_size=args.leapfrog_step_size,
                 init=args.init,
                 method=args.method,
                 parameters_tolerance=args.parameters_tolerance,
                 parameters_initialization=parameters_init_string,
                 chain_length=args.chain_length,
                 max_parameters_iterations=args.max_parameters_iterations,
                 random_seed=args.random_seed,
                 input_file=args.input_csv_file)

    if args.predictors_file:
        attrs['predictors_file'] = args.predictors_file

    if args.random_state is None:
        attrs['random_state'] = 'None'
    else:
        attrs['random_state'] = args.random_state

    models = {o: [] for o in outcome_names}
    for i, f in enumerate(outcome_names):
        for j, eps_gamma in enumerate(epsilon_gammas):
            for k, eps_theta in enumerate(epsilon_thetas):
                run_result = run_femh1_linear_mcmc(
                    outcomes[f], predictors, n_components=args.n_components,
                    n_chains=args.n_init, epsilon_theta=eps_theta,
                    epsilon_gamma=eps_gamma,
                    sigma_theta=args.sigma_theta,
                    sigma_gamma=args.sigma_gamma,
                    n_leapfrog_steps=args.n_leapfrog_steps,
                    leapfrog_step_size=args.leapfrog_step_size,
                    init=args.init, method=args.method,
                    parameters_tolerance=args.parameters_tolerance,
                    parameters_initialization=args.parameters_initialization,
                    chain_length=args.chain_length,
                    max_parameters_iterations=args.max_parameters_iterations,
                    verbose=args.verbose, random_seed=args.random_seed,
                    random_state=args.random_state)

                chains = create_model_dict(f, eps_gamma, eps_theta, times,
                                           predictor_names, run_result, attrs)

                models[f].append(chains)

    if args.output_file:
        with shelve.open(args.output_file) as db:
            for f in outcome_names:
                if f in db:
                    db[f] += models[f]
                else:
                    db[f] = models[f]


if __name__ == '__main__':
    main()
