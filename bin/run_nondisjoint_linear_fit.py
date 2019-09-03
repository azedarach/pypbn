import argparse
import datetime
import numpy as np
import pypbn
import time
import xarray as xr

from pypbn import FEMBVBinLinear


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


def generate_fit_inputs(times, categorical_values, memory=0,
                        double_indicators=False):
    lagged_times = times[memory:]

    fields = [f for f in categorical_values]
    lagged_outcomes = {f: categorical_values[f][memory:]
                       for f in categorical_values}

    n_samples = lagged_times.shape[0]
    n_features = len(lagged_outcomes)

    if memory == 0:
        # intercept term only
        predictors = np.ones((1, n_samples), dtype='i8')
        predictor_names = ['unresolved']
    else:
        if double_indicators:
            n_indicators = 2 * n_features * memory
        else:
            n_indicators = n_features * memory

        predictor_names = ['unresolved']
        predictors = np.zeros((1 + n_indicators, n_samples), dtype='i8')
        predictors[0, :] = 1

        index = 1
        for lag in range(1, memory + 1):
            for i, f in enumerate(fields):
                lagged_field = categorical_values[f][memory - lag:-lag]
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


def run_fembv_linear_fit(Y, X, n_components=None, n_init=1,
                         epsilon=0, max_tv_norm=None,
                         init=None, tolerance=1e-6, parameters_tolerance=1e-6,
                         parameters_initialization=None, max_iterations=1000,
                         max_parameters_iterations=1000,
                         max_affiliations_iterations=1000,
                         verbose=0, random_seed=0, random_state=None):
    best_cost = None
    best_affiliations = None
    best_parameters = None
    best_log_likelihood_bound = None
    best_n_iter = None
    best_runtime = None

    success = False
    for i in range(n_init):
        start_time = time.perf_counter()

        model = pypbn.FEMBVBinLinear(n_components=n_components,
                                     epsilon=epsilon,
                                     max_tv_norm=max_tv_norm,
                                     init=init,
                                     tolerance=tolerance,
                                     parameters_tolerance=parameters_tolerance,
                                     parameters_initialization=parameters_initialization,
                                     max_iterations=max_iterations,
                                     max_parameters_iterations=max_parameters_iterations,
                                     max_affiliations_iterations=max_affiliations_iterations,
                                     verbose=verbose, random_seed=random_seed,
                                     random_state=random_state)

        affiliations = model.fit_transform(Y, X)
        parameters = model.parameters_
        cost = model.cost_
        log_likelihood_bound = model.log_likelihood_bound_
        n_iter = model.n_iter_

        end_time = time.perf_counter()

        runtime = end_time - start_time

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_affiliations = affiliations.copy()
            best_parameters = parameters.copy()
            best_log_likelihood_bound = log_likelihood_bound
            best_n_iter = n_iter
            best_runtime = runtime
            success = True

    if not success:
        raise RuntimeError('failed to fit FEM-BV model')

    return dict(affiliations=best_affiliations, parameters=best_parameters,
                cost=best_cost, log_likelihood_bound=best_log_likelihood_bound,
                n_iter=best_n_iter, runtime=best_runtime)


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


def write_output_file(output_file, outcome_names, predictor_names,
                      max_tv_norms, regularizations,
                      n_components, times, affiliations, parameters, n_iter,
                      runtime, cost, log_likelihood_bound, attrs):
    component_indices = np.arange(n_components)

    affiliations_da = xr.DataArray(
        affiliations,
        coords={'outcome': outcome_names,
                'max_tv_norm': max_tv_norms,
                'regularization': regularizations,
                'component': component_indices,
                'time': times},
        dims=['outcome', 'max_tv_norm', 'regularization',
              'component', 'time'])
    parameters_da = xr.DataArray(
        parameters,
        coords={'outcome': outcome_names,
                'max_tv_norm': max_tv_norms,
                'regularization': regularizations,
                'component': component_indices,
                'predictor': predictor_names},
        dims=['outcome', 'max_tv_norm', 'regularization',
              'component', 'predictor'])
    n_iter_da = xr.DataArray(
        n_iter,
        coords={'outcome': outcome_names,
                'max_tv_norm': max_tv_norms,
                'regularization': regularizations},
        dims=['outcome', 'max_tv_norm', 'regularization'])
    runtime_da = xr.DataArray(
        runtime,
        coords={'outcome': outcome_names,
                'max_tv_norm': max_tv_norms,
                'regularization': regularizations},
        dims=['outcome', 'max_tv_norm', 'regularization'])
    cost_da = xr.DataArray(
        cost,
        coords={'outcome': outcome_names,
                'max_tv_norm': max_tv_norms,
                'regularization': regularizations},
        dims=['outcome', 'max_tv_norm', 'regularization'])
    log_like_da = xr.DataArray(
        log_likelihood_bound,
        coords={'outcome': outcome_names,
                'max_tv_norm': max_tv_norms,
                'regularization': regularizations},
        dims=['outcome', 'max_tv_norm', 'regularization'])

    ds = xr.Dataset(data_vars={'affiliations': affiliations_da,
                               'parameters': parameters_da,
                               'n_iter': n_iter_da,
                               'runtime': runtime_da,
                               'cost': cost_da,
                               'log_likelihood_bound': log_like_da},
                    coords={'outcome': outcome_names,
                            'max_tv_norm': max_tv_norms,
                            'regularization': regularizations,
                            'component': component_indices,
                            'time': times,
                            'predictor': predictor_names},
                    attrs=attrs)

    ds.to_netcdf(output_file)


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Run FEM-BV-binary fit on categorical data')

    parser.add_argument('input_csv_file',
                        help='file containing input data')
    parser.add_argument('--n-components', dest='n_components',
                        type=int, default=1,
                        help='number of clusters')
    parser.add_argument('--regularization', dest='regularization',
                        type=float, action='append',
                        help='regularization parameter')
    parser.add_argument('--max-tv-norm', dest='max_tv_norm',
                        type=float, action='append',
                        help='maximum TV norm')
    parser.add_argument('--memory', dest='memory', type=int,
                        default=0, help='maximum lag')
    parser.add_argument('--n-init', dest='n_init', type=int,
                        default=1, help='number of initializations')
    parser.add_argument('--tolerance', dest='tolerance',
                        type=float, default=1e-6,
                        help='stopping criterion tolerance')
    parser.add_argument('--init', dest='init',
                        choices=['random'], default='random',
                        help='initialization method')
    parser.add_argument('--parameters-tolerance', dest='parameters_tolerance',
                        type=float, default=1e-6,
                        help='parameters optimization tolerance')
    parser.add_argument('--parameters-initialization',
                        dest='parameters_initialization',
                        choices=[pypbn.Uniform, pypbn.Random, pypbn.Current],
                        default=pypbn.Uniform,
                        help='parameters optimization initial guess method')
    parser.add_argument('--max-iterations', dest='max_iterations',
                        type=int, default=10000,
                        help='maximum number of iterations')
    parser.add_argument('--max-parameters-iterations',
                        dest='max_parameters_iterations',
                        type=int, default=10000,
                        help='maximum number of parameter optimizer iterations')
    parser.add_argument('--max-affiliations-iterations',
                        dest='max_affiliations_iterations',
                        type=int, default=10000,
                        help='maximum number of affiliations optimizer iterations')
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

    times, outcomes, predictors, predictor_names = generate_fit_inputs(
        times, categorical_values, memory=args.memory,
        double_indicators=args.double_indicators)

    if args.predictors_file:
        write_predictors(args.predictors_file, times, predictors,
                         is_daily_data=is_daily_data,
                         predictor_names=predictor_names)

    if not args.regularization:
        regularizations = [0]
    else:
        regularizations = args.regularization

    if not args.max_tv_norm:
        max_tv_norms = [-1]
    else:
        max_tv_norms = args.max_tv_norm

    outcome_names = [f for f in outcomes]

    n_outcomes = len(outcome_names)
    n_regularizations = np.size(regularizations)
    n_max_tv_norms = np.size(max_tv_norms)
    n_features, n_samples = predictors.shape
    component_indices = np.arange(args.n_components)

    affiliations = np.empty((n_outcomes,  n_max_tv_norms, n_regularizations,
                             args.n_components, n_samples), dtype='f8')
    parameters = np.empty((n_outcomes, n_max_tv_norms, n_regularizations,
                           args.n_components, n_features), dtype='f8')
    n_iter = np.empty((n_outcomes, n_max_tv_norms, n_regularizations),
                      dtype='i8')
    runtime = np.empty((n_outcomes, n_max_tv_norms, n_regularizations),
                       dtype='f8')
    cost = np.empty((n_outcomes, n_max_tv_norms, n_regularizations),
                    dtype='f8')
    log_like = np.empty((n_outcomes, n_max_tv_norms, n_regularizations),
                        dtype='f8')

    for i, f in enumerate(outcomes):
        for j, c in enumerate(max_tv_norms):
            for k, epsilon in enumerate(regularizations):
                fit_result = run_fembv_linear_fit(
                    outcomes[f], predictors, n_components=args.n_components,
                    n_init=args.n_init, epsilon=epsilon, max_tv_norm=c,
                    init=args.init, tolerance=args.tolerance,
                    parameters_tolerance=args.parameters_tolerance,
                    parameters_initialization=args.parameters_initialization,
                    max_iterations=args.max_iterations,
                    max_parameters_iterations=args.max_parameters_iterations,
                    max_affiliations_iterations=args.max_affiliations_iterations,
                    verbose=args.verbose, random_seed=args.random_seed,
                    random_state=args.random_state)

                affiliations[i, j, k, ...] = fit_result['affiliations']
                parameters[i, j, k, ...] = fit_result['parameters']
                n_iter[i, j, k] = fit_result['n_iter']
                runtime[i, j, k] = fit_result['runtime']
                cost[i, j, k] = fit_result['cost']
                log_like[i, j, k] = fit_result['log_likelihood_bound']

    if args.output_file:
        attrs = dict(n_components=args.n_components,
                     n_init=args.n_init,
                     init=args.init,
                     tolerance=args.tolerance,
                     parameters_tolerance=args.parameters_tolerance,
                     max_iterations=args.max_iterations,
                     max_parameters_iterations=args.max_parameters_iterations,
                     max_affiliations_iterations=args.max_affiliations_iterations,
                     random_seed=args.random_seed,
                     input_file=args.input_csv_file)

        if args.predictors_file:
            attrs['predictors_file'] = args.predictors_file

        if args.random_state is None:
            attrs['random_state'] = 'None'
        else:
            attrs['random_state'] = args.random_state

        write_output_file(args.output_file, outcome_names=outcome_names,
                          predictor_names=predictor_names,
                          regularizations=regularizations,
                          max_tv_norms=max_tv_norms,
                          n_components=args.n_components,
                          affiliations=affiliations,
                          parameters=parameters, n_iter=n_iter,
                          runtime=runtime, cost=cost,
                          log_likelihood_bound=log_like, attrs=attrs)


if __name__ == '__main__':
    main()
