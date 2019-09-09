import argparse
import datetime
import numpy as np
import re
import shelve

from sklearn.utils import check_random_state


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
                    if len(c[p] > 1):
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
                    if len(c[p] > 1):
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

    for model in models:
        if best_model is None:
            best_model = model
            continue

        if criterion == 'cost':
            if model['cost'] < best_model['cost']:
                best_model = model
        elif criterion == 'bic':
            best_model_bic = get_model_bic(model, threshold=threshold,
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
    times = models[0]['time']
    for i in range(1, n_models):
        if not np.all(models[i]['time'] == times):
            raise RuntimeError('times do not match in all models')
    return times


def get_maximum_lag(models):
    max_lag = None
    for m in models:
        for c in m['components']:
            predictor_names = [p for p in c if p != 'unresolved']
            for p in predictor_names:
                involved_indices = parse_predictor_name(p)
                for index in involved_indices:
                    if max_lag is None or index[-1] > max_lag:
                        max_lag = index[-1]
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


def get_indicator_value(index_name, index_phase, index_lag, current_indices):
    if index_name not in current_indices:
        raise ValueError('data for index %s not found' % index_name)

    if index_lag == 0:
        raise ValueError('lag must be greater than 0')

    lagged_value = current_indices[index_name][-index_lag]

    if np.sign(lagged_value) == np.sign(index_phase):
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


def write_simulated_indices(output_file, times, indices, is_daily=False):
    index_names = [index for index in indices]

    if is_daily:
        fields = ([('year', '%d'), ('month', '%d'), ('day', '%d')] +
                  [(index, '%d') for index in index_names])
        time_cols = 3
    else:
        fields = ([('year', '%d'), ('month', '%d')] +
                  [(index, '%d') for index in index_names])
        time_cols = 2

    header = ','.join([f[0] for f in fields])
    fmt = ','.join([f[1] for f in fields])

    n_fields = len(fields)
    n_samples = np.size(times)

    data = np.empty((n_samples, n_fields), dtype='i8')
    data[:, 0] = np.array([dt.year for dt in times], dtype='i8')
    data[:, 1] = np.array([dt.month for dt in times], dtype='i8')
    if is_daily:
        data[:, 2] = np.array([dt.day for dt in times], dtype='i8')

    for i, index in enumerate(index_names):
        data[:, time_cols + i] = indices[index]

    np.savetxt(output_file, data, header=header, fmt=fmt)


def print_simulated_indices(times, indices, is_daily=False):
    index_names = [index for index in indices]

    if is_daily:
        fields = ([('year', '{:d}'), ('month', '{:d}'), ('day', '{:d}')] +
                  [(index, '{:d}') for index in index_names])
        time_cols = 3
    else:
        fields = ([('year', '{:d}'), ('month', '{:d}')] +
                  [(index, '{:d}') for index in index_names])
        time_cols = 2

    header = ','.join([f[0] for f in fields])
    fmt = ','.join([f[1] for f in fields])

    n_samples = np.size(times)

    print('# ' + header)

    for i in range(n_samples):
        values = [times[i].year, times[i].month]
        if is_daily:
            values.append(times[i].day)
        for index in index_names:
            values.append(indices[index][i])

        print(fmt.format(*values))


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Simulate data from linear non-disjoint model')

    parser.add_argument('models_db', help='database of models')
    parser.add_argument('--output-file', dest='output_file', default='',
                        help='name of output file to write to')
    parser.add_argument('--random-seed', dest='random_seed', default=None,
                        type=int, help='random seed to use')

    args = parser.parse_args()

    return args


def main():
    args = parse_cmd_line_args()

    random_state = np.random.RandomState(args.random_seed)

    models = read_models(args.models_db)

    index_names = get_required_index_names(models)
    model_times = get_simulation_times(models)
    max_lag = get_maximum_lag(models)

    time_step = model_times[1] - model_times[0]
    if time_step > datetime.timedelta(days=1):
        is_daily = False
    else:
        is_daily = True

    simulated_times, simulated_indices = generate_initial_conditions(
        index_names, model_times[0], max_lag, is_daily=is_daily,
        random_state=random_state)

    simulated_times, simulated_indices = simulate_indices(
        models, initial_times=simulated_times,
        initial_indices=simulated_indices,
        is_daily=is_daily, random_state=random_state)

    if args.output_file:
        write_simulated_indices(args.output_file, simulated_times,
                                simulated_indices, is_daily=is_daily)
    else:
        print_simulated_indices(simulated_times, simulated_indices,
                                is_daily=is_daily)


if __name__ == '__main__':
    main()
