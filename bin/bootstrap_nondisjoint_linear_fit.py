import argparse
import datetime
import numpy as np
import pypbn
import re
import shelve
import time

from math import pi
from sklearn.utils import check_random_state

from pypbn import FEMBVBinLinear


def calculate_bic(log_likelihood, n_pars, n_samples):
    bic = -2 * log_likelihood + n_pars * (np.log(n_samples) - np.log(2 * pi))
    return bic


def get_model_bic(model, threshold=5e-8, cluster_bic=True):
    log_like = model['log_likelihood_bound']

    gamma = model['affiliations']
    components = model['components']

    n_components, n_samples = gamma.shape
    n_parameters = 0

    if cluster_bic:
        n_parameters += (n_components - 1)
        n_parameters += np.sum(np.abs(np.diff(gamma, axis=1)))
        if threshold > 0:
            for c in components:
                predictor_names = [p for p in c]
                for p in predictor_names:
                    if len(c[p]) > 1:
                        theta = c[p][0]
                    else:
                        theta = c[p]
                    if np.abs(theta) > threshold:
                        n_parameters += 1
        else:
            for c in components:
                predictor_names = [p for p in c]
                for p in predictor_names:
                    n_parameters += 1
    else:
        if threshold > 0:
            n_parameters += np.sum(np.abs(gamma) > threshold)
            for c in components:
                predictor_names = [p for p in c]
                for p in predictor_names:
                    if len(c[p]) > 1:
                        theta = c[p][0]
                    else:
                        theta = c[p]
                    if np.abs(theta) > threshold:
                        n_parameters += 1
        else:
            n_parameters += np.size(gamma) - n_samples
            for c in components:
                predictor_names = [p for p in c]
                for p in predictor_names:
                    n_parameters += 1

    return calculate_bic(log_like, n_parameters, n_samples)


def get_best_model(models, criterion='bic', threshold=5e-8, cluster_bic=True):
    best_model = None

    if isinstance(models, dict):
        best_model = models
        return best_model

    for model in models:
        if best_model is None:
            best_model = model
            continue

        if criterion == 'cost':
            if model['cost'] < best_model['cost']:
                best_model = model
        elif criterion == 'bic':
            best_model_bic = get_model_bic(best_model, threshold=threshold,
                                           cluster_bic=cluster_bic)
            model_bic = get_model_bic(model, threshold=threshold,
                                      cluster_bic=cluster_bic)
            if model_bic < best_model_bic:
                best_model = model
        else:
            raise ValueError('invalid criterion (criterion=%r)' % criterion)

    return best_model


def read_models(models_db, criterion='cost'):
    models = []

    with shelve.open(models_db, 'r') as db:
        outcomes = [f for f in db]
        for f in outcomes:
            best_model = get_best_model(db[f], criterion=criterion)
            models.append(best_model)

    return models


def parse_predictor_name(predictor_name):
    lagged_index_sign_pattern = '(.*)_(positive|negative)_lag_([0-9]+)'
    match = re.match(lagged_index_sign_pattern, predictor_name)
    if not match:
        raise RuntimeError('could not parse predictor name')
    else:
        index_name = match.group(1)
        index_phase = 1 if match.group(2) == 'positive' else -1
        index_lag = int(match.group(3))
        return [(index_name, index_phase, index_lag)]


def get_required_index_names(models):
    index_names = [m['outcome'] for m in models]
    for m in models:
        for c in m['components']:
            predictor_names = [p for p in c if p != 'unresolved']
            for p in predictor_names:
                involved_indices = parse_predictor_name(p)
                for index in involved_indices:
                    if index[0] not in index_names:
                        index_names.append(index[0])
    return index_names


def get_simulation_times(models):
    n_models = len(models)

    start_time = None
    end_time = None

    for i in range(n_models):
        model_start_time = np.min(models[i]['time'])
        model_end_time = np.max(models[i]['time'])

        if start_time is None or model_start_time > start_time:
            start_time = model_start_time

        if end_time is None or model_end_time < end_time:
            end_time = model_end_time

    times = None
    masked_models = models.copy()

    for i in range(n_models):
        model_times = models[i]['time']
        mask = np.logical_and(model_times >= start_time,
                              model_times <= end_time)
        valid_model_times = model_times[mask]
        if times is None:
            times = valid_model_times
        else:
            if not np.all(valid_model_times == times):
                raise RuntimeError('models do not have same timepoints')

        masked_models[i]['time'] = valid_model_times

    return times, masked_models


def get_model_lags(model):
    lags = []
    for c in model['components']:
        predictor_names = [p for p in c if p != 'unresolved']
        for p in predictor_names:
            involved_indices = parse_predictor_name(p)
            for index in involved_indices:
                if index[-1] not in lags:
                    lags.append(index[-1])

    return sorted(lags)


def get_maximum_lag(models):
    max_lag = None
    for m in models:
        lags = get_model_lags(m)
        for lag in lags:
            if max_lag is None or lag > max_lag:
                max_lag = lag
    return max_lag


def generate_initial_conditions(index_names, start_time, max_lag,
                                is_daily=False, random_state=None):
    rng = check_random_state(random_state)

    if is_daily:
        t0 = start_time - datetime.timedelta(days=1)
    else:
        if start_time.month == 1:
            t0 = start_time.replace(month=12,
                                    year=(start_time.year - 1))
        else:
            t0 = start_time.replace(month=(start_time.month - 1))

    times = [t0]
    for i in range(1, max_lag):
        if is_daily:
            t = times[i - 1] - datetime.timedelta(days=1)
        else:
            if times[i - 1].month == 1:
                t = times[i - 1].replace(month=12, year=(times[i - 1].year - 1))
            else:
                t = times[i - 1].replace(month=(times[i - 1].month - 1))
        times.append(t)

    times = times[::-1]
    n_times = len(times)

    index_values = {index: [] for index in index_names}
    for i in range(n_times):
        for index in index_values:
            index_values[index].append(rng.choice([-1, 1]))

    return times, index_values


def get_affiliations_at_time(model, t):
    times = model['time']
    affiliations = model['affiliations']

    pos = np.nonzero(times == t)

    return affiliations[:, pos].ravel()


def get_indicator_value(index_name, index_phase, index_lag, current_indices,
                        zero_positive=False):
    if index_name not in current_indices:
        raise ValueError('data for index %s not found' % index_name)

    if index_lag == 0:
        raise ValueError('lag must be greater than 0')

    lagged_value = current_indices[index_name][-index_lag]

    lagged_sign = np.sign(lagged_value)
    if lagged_sign == 0 and zero_positive:
        lagged_sign = 1
    elif lagged_sign == 0:
        lagged_sign = -1

    if lagged_sign == np.sign(index_phase):
        return 1
    else:
        return 0


def get_predictor_value(predictor_name, current_indices):
    involved_indices = parse_predictor_name(predictor_name)

    value = 1
    for index in involved_indices:
        indicator_value = get_indicator_value(index[0], index[1], index[2],
                                              current_indices)
        value *= indicator_value

    return value


def sample_from_model(model, t, current_indices, random_state=None):
    rng = check_random_state(random_state)
    gamma = get_affiliations_at_time(model, t)

    parameters = {}
    for i, c in enumerate(model['components']):
        predictor_names = [p for p in c]
        for p in predictor_names:
            if len(c[p]) > 1:
                theta = c[p][0]
            else:
                theta = c[p]
            if p not in parameters:
                parameters[p] = gamma[i] * theta
            else:
                parameters[p] += gamma[i] * theta

    outcome_prob = 0
    if 'unresolved' in parameters:
        outcome_prob += parameters['unresolved']

    for p in parameters:
        if p == 'unresolved':
            continue

        predictor_value = get_predictor_value(p, current_indices)
        outcome_prob += parameters[p] * predictor_value

    u = rng.uniform()
    if u <= outcome_prob:
        return 1
    else:
        return -1


def simulate_indices(models, initial_times, initial_indices,
                     is_daily=False, random_state=None):
    rng = check_random_state(random_state)

    modelled_indices = [m['outcome'] for m in models]
    unmodelled_indices = [index for index in initial_indices
                          if index not in modelled_indices]

    n_initial_conditions = np.size(initial_times)
    simulated_indices = initial_indices

    n_samples = None
    simulated_times = None
    for m in models:
        if n_samples is None:
            n_samples = np.size(m['time'])
            simulated_times = np.concatenate([initial_times, m['time']])
        else:
            if np.size(m['time']) != n_samples:
                raise RuntimeError(
                    'models do not have same number of time points')

    simulated_indices = initial_indices.copy()
    for sample_index in range(n_samples):
        t = simulated_times[sample_index + n_initial_conditions]
        for m in models:
            index = m['outcome']
            value = sample_from_model(m, t, simulated_indices, rng)
            simulated_indices[index].append(value)
        for index in unmodelled_indices:
            value = rng.choice([-1, 1])
            simulated_indices[index].append(value)

    return simulated_times, simulated_indices


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


def get_predictor_values(predictor_name, categorical_values, start_index):
    involved_indices = parse_predictor_name(predictor_name)
    for index in involved_indices:
        if index[0] not in categorical_values:
            raise RuntimeError('no data found for index %s' % index[0])

    n_samples = np.size(categorical_values[involved_indices[0][0]])

    predictor_values = []
    for i in range(start_index, n_samples):
        current_indices = {f: categorical_values[f][:i]
                           for f in categorical_values}
        value = get_predictor_value(predictor_name, current_indices)
        predictor_values.append(value)

    return np.asarray(predictor_values, dtype='i8')


def generate_fit_inputs(model, times, categorical_values):
    lags = get_model_lags(model)
    if not lags:
        lags = [0]

    max_lag = np.max(lags)
    n_lags = np.size(lags)
    outcome = model['outcome']

    lagged_times = times[max_lag:]
    lagged_outcomes = categorical_values[outcome][max_lag:]

    predictor_names = []
    for c in model['components']:
        for p in c:
            if p not in predictor_names:
                predictor_names.append(p)

    n_predictors = len(predictor_names)
    n_samples = np.size(lagged_times)
    predictors = np.empty((n_predictors, n_samples), dtype='i8')

    for i, p in enumerate(predictor_names):
        if p == 'unresolved':
            predictors[i, :] = 1
        else:
            predictors[i, :] = get_predictor_values(p, categorical_values,
                                                    start_index=max_lag)

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


def create_model_dict(outcome, max_tv_norm, regularization,
                      times, predictor_names, fit_result, attrs=None):
    n_components = fit_result['affiliations'].shape[0]
    model = {'outcome': outcome,
             'max_tv_norm': max_tv_norm,
             'regularization': regularization,
             'n_components': n_components,
             'affiliations': fit_result['affiliations'].copy(),
             'time': times,
             'cost': fit_result['cost'],
             'log_likelihood_bound': fit_result['log_likelihood_bound'],
             'n_iter': fit_result['n_iter'],
             'runtime': fit_result['runtime']
             }

    components = []
    for i in range(n_components):
        component_parameters = fit_result['parameters'][i]
        component = {p: (component_parameters[pi], None, None)
                     for pi, p in enumerate(predictor_names)}
        components.append(component)

    model['components'] = components

    if attrs is not None:
        for attr in attrs:
            model[attr] = attrs[attr]

    return model


def summarise_models(bootstrapped_models, alpha=0.05):
    summarised_models = {f: None for f in bootstrapped_models}

    for f in bootstrapped_models:
        n_models = len(bootstrapped_models[f])

        summarised_models[f] = dict(
            outcome=f,
            max_tv_norm=bootstrapped_models[f][0]['max_tv_norm'],
            regularization=bootstrapped_models[f][0]['regularization'],
            n_components=bootstrapped_models[f][0]['n_components'],
            n_init=bootstrapped_models[f][0]['n_init'],
            init=bootstrapped_models[f][0]['init'],
            tolerance=bootstrapped_models[f][0]['tolerance'],
            parameters_tolerance=bootstrapped_models[f][0]['parameters_tolerance'],
            max_iterations=bootstrapped_models[f][0]['max_iterations'],
            parameters_initialization=bootstrapped_models[f][0]['parameters_initialization'],
            max_parameters_iterations=bootstrapped_models[f][0]['max_parameters_iterations'],
            max_affiliations_iterations=bootstrapped_models[f][0]['max_affiliations_iterations'],
            random_seed=bootstrapped_models[f][0]['random_seed'],
            time=bootstrapped_models[f][0]['time'])

        log_like = []
        cost = []
        n_iter = []
        runtime = []

        components = []
        for c in bootstrapped_models[f][0]['components']:
            predictor_names = [p for p in c]
            component = {p: [] for p in predictor_names}
            components.append(component)

        n_components, n_samples = bootstrapped_models[f][0]['affiliations'].shape
        affiliations = np.empty((n_components, n_samples, n_models))

        for i in range(n_models):
            log_like.append(bootstrapped_models[f][i]['log_likelihood_bound'])
            cost.append(bootstrapped_models[f][i]['cost'])
            n_iter.append(bootstrapped_models[f][i]['n_iter'])
            runtime.append(bootstrapped_models[f][i]['runtime'])

            for i, c in enumerate(bootstrapped_models[f][i]['components']):
                predictor_names = [p for p in c]
                for p in predictor_names:
                    if len(c[p]) > 1:
                        theta = c[p][0]
                    else:
                        theta = c[p]
                    components[i][p].append(theta)

            affiliations[:, :, i] = bootstrapped_models[f][i]['affiliations']

        tail_area = 0.5 * alpha
        lower_index = max(0, int(np.floor(tail_area * n_models)))
        upper_index = min(int(np.ceil((1 - tail_area) * n_models)), n_models - 1)

        log_like = np.sort(np.asarray(log_like))
        cost = np.sort(np.asarray(cost))
        n_iter = np.sort(np.asarray(n_iter))
        runtime = np.sort(np.asarray(runtime))

        summarised_models[f]['log_likelihood_bound'] = (
            np.mean(log_like), log_like[lower_index], log_like[upper_index])
        summarised_models[f]['cost'] = (
            np.mean(cost), cost[lower_index], cost[upper_index])
        summarised_models[f]['n_iter'] = (
            np.mean(n_iter), n_iter[lower_index], n_iter[upper_index])
        summarised_models[f]['runtime'] = (
            np.mean(runtime), runtime[lower_index], runtime[upper_index])

        for i, c in enumerate(components):
            predictor_names = [p for p in c]
            for p in predictor_names:
                predictor_coeffs = np.sort(np.asarray(components[i][p]))
                mean_coeff = np.mean(predictor_coeffs)
                lower_bnd = predictor_coeffs[lower_index]
                upper_bnd = predictor_coeffs[upper_index]
                components[i][p] = (mean_coeff, lower_bnd, upper_bnd)

        summarised_models[f]['components'] = components

        affiliations = np.sort(affiliations, axis=-1)
        affiliations_bnds = np.empty((n_components, n_samples, 3))
        affiliations_bnds[:, :, 0] = np.mean(affiliations, axis=-1)
        affiliations_bnds[:, :, 1] = affiliations[:, :, lower_index]
        affiliations_bnds[:, :, 2] = affiliations[:, :, upper_index]

        summarised_models[f]['affiliations'] = affiliations_bnds

    return summarised_models


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Run parametric bootstrap fits from fitted models')

    parser.add_argument('models_db', help='database of models')
    parser.add_argument('--output-database', dest='output_db', default='',
                        help='name of output database to write to')
    parser.add_argument('--n-bootstrap', dest='n_bootstrap', default=200,
                        type=int, help='number of bootstrap samples')
    parser.add_argument('--random-seed', dest='random_seed', default=None,
                        type=int, help='random seed to use')

    args = parser.parse_args()

    return args


def main():
    args = parse_cmd_line_args()

    random_state = np.random.RandomState(args.random_seed)

    models = read_models(args.models_db)

    index_names = get_required_index_names(models)
    model_times, models = get_simulation_times(models)
    max_lag = get_maximum_lag(models)

    time_step = model_times[1] - model_times[0]
    if time_step > datetime.timedelta(days=1):
        is_daily = False
    else:
        is_daily = True

    bootstrapped_models = {m['outcome']: [] for m in models}
    for i in range(args.n_bootstrap):
        simulated_times, simulated_indices = generate_initial_conditions(
            index_names, model_times[0], max_lag, is_daily=is_daily,
            random_state=random_state)

        simulated_times, simulated_indices = simulate_indices(
            models, initial_times=simulated_times,
            initial_indices=simulated_indices,
            is_daily=is_daily, random_state=random_state)

        categorical_values = to_categorical_values(simulated_indices)

        for model in models:
            lags = get_model_lags(model)
            outcome = model['outcome']

            model_times, model_outcomes, model_predictors, model_predictor_names  = \
                generate_fit_inputs(model, simulated_times, categorical_values)

            if model['parameters_initialization'] == 'Uniform':
                parameters_initialization = pypbn.Uniform
            elif model['parameters_initialization'] == 'Random':
                parameters_initialization = pypbn.Random
            elif model['parameters_initialization'] == 'Current':
                parameters_initialization = pypbn.Current
            else:
                raise ValueError('invalid initialization %r' %
                                 model['parameters_initialization'])

            attrs = dict(n_components=model['n_components'],
                         n_init=model['n_init'],
                         init=model['init'],
                         tolerance=model['tolerance'],
                         parameters_tolerance=model['parameters_tolerance'],
                         max_iterations=model['max_iterations'],
                         parameters_initialization=model['parameters_initialization'],
                         max_parameters_iterations=model['max_parameters_iterations'],
                         max_affiliations_iterations=model['max_affiliations_iterations'],
                         random_seed=model['random_seed'])

            fit_result = run_fembv_linear_fit(
                model_outcomes, model_predictors,
                n_components=model['n_components'],
                n_init=model['n_init'],
                epsilon=model['regularization'],
                max_tv_norm=model['max_tv_norm'],
                init=model['init'],
                tolerance=model['tolerance'],
                parameters_tolerance=model['parameters_tolerance'],
                parameters_initialization=parameters_initialization,
                max_iterations=model['max_iterations'],
                max_parameters_iterations=model['max_parameters_iterations'],
                max_affiliations_iterations=model['max_affiliations_iterations'],
                random_seed=model['random_seed'],
                random_state=random_state)

            fitted_model = create_model_dict(
                model['outcome'], model['max_tv_norm'], model['regularization'],
                model_times, model_predictor_names,
                fit_result, attrs)

            bootstrapped_models[model['outcome']].append(fitted_model)

    summarised_models = summarise_models(bootstrapped_models)

    if args.output_db:
        with shelve.open(args.output_db) as db:
            for f in summarised_models:
                if isinstance(summarised_models[f], dict):
                    db[f] = summarised_models[f]
                else:
                    if f in db:
                        db[f] += summarised_models[f]
                    else:
                        db[f] = summarised_models[f]


if __name__ == '__main__':
    main()
