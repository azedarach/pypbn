from __future__ import division

import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils import check_random_state


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


def get_index_signs(index_values, zero_positive=False):
    index_signs = np.sign(index_values)
    if zero_positive:
        index_signs[index_signs == 0] = 1
    else:
        index_signs[index_signs == 0] = -1

    return index_signs


def calculate_phase_fractions(data, zero_positive=False):
    index_names = [f for f in data]
    n_indices = len(index_names)
    pos_fractions = np.full((n_indices,), np.NaN)

    for i, index in enumerate(index_names):
        pos_count = get_index_signs(data[index], zero_positive=zero_positive)
        pos_count[pos_count < 0] = 0

        pos_fractions[i] = np.sum(pos_count) / np.size(pos_count)

    return {index_names[i]: pos_fractions[i] for i in range(n_indices)}


def get_positive_phase_duration_frequencies(data, zero_positive=False):
    index_names = [f for f in data]
    n_indices = len(index_names)

    pos_duration_frequencies = {f: [] for f in index_names}
    for index in index_names:
        index_signs = get_index_signs(data[index], zero_positive=zero_positive)
        n_samples = np.size(index_signs)

        pos_state_durations = []
        if index_signs[0] > 0:
            current_duration = 1
        else:
            current_duration = 0

        for i in range(1, n_samples):
            if index_signs[i] > 0 and index_signs[i - 1] > 0:
                current_duration += 1
            elif index_signs[i] <= 0 and index_signs[i - 1] > 0:
                pos_state_durations.append(current_duration)
                current_duration = 0
            elif index_signs[i] > 0 and index_signs[i - 1] <= 0:
                current_duration = 1

            if i == n_samples - 1 and index_signs[i] > 0:
                pos_state_durations.append(current_duration)

        pos_state_durations = np.asarray(pos_state_durations, dtype='i8')
        pos_duration_frequencies[index] = np.bincount(pos_state_durations)

    return pos_duration_frequencies


def get_positive_phase_duration_summary(pos_phase_durations):
    index_names = [f for f in pos_phase_durations]

    summaries = {f: {'minimum': None,
                     'lower_quartile': None,
                     'mode': None,
                     'mean': None,
                     'median': None,
                     'upper_quartile': None,
                     'maximum': None} for f in index_names}

    for index in index_names:
        index_pos_phase_durations = pos_phase_durations[index]
        n_pos_phases = np.sum(index_pos_phase_durations)

        observed_durations = np.nonzero(index_pos_phase_durations)
        observed_duration_counts = index_pos_phase_durations[observed_durations]
        observed_duration_propns = observed_duration_counts / n_pos_phases
        observed_durations = observed_durations[0]

        mode_index = np.argmax(observed_duration_counts)

        cumulative_counts = np.cumsum(observed_duration_counts)
        cumulative_propns = cumulative_counts / n_pos_phases
        lq_index = np.min(np.nonzero(cumulative_propns >= 0.25)[0])
        median_index = np.min(np.nonzero(cumulative_propns >= 0.5)[0])
        uq_index = np.min(np.nonzero(cumulative_propns >= 0.75)[0])

        summaries[index]['minimum'] = np.min(observed_durations)
        summaries[index]['lower_quartile'] = observed_durations[lq_index]
        summaries[index]['mode'] = observed_durations[mode_index]
        summaries[index]['mean'] = np.sum(
            observed_duration_propns * observed_durations)
        summaries[index]['median'] = observed_durations[median_index]
        summaries[index]['upper_quartile'] = observed_durations[uq_index]
        summaries[index]['maximum'] = np.max(observed_durations)

    return summaries


def geometric_pmf(x, p):
    return p * (1 - p) ** (x - 1)


def get_positive_phase_duration_geometric_fits(pos_phase_durations,
                                               n_bootstrap=1000,
                                               alpha = 0.05,
                                               random_state=None):
    rng = check_random_state(random_state)

    index_names = [f for f in pos_phase_durations]

    geometric_fits = {f: {'mle': None} for f in index_names}

    for index in index_names:
        index_pos_phase_durations = pos_phase_durations[index]
        observed_durations = np.nonzero(index_pos_phase_durations)
        observed_counts = index_pos_phase_durations[observed_durations]
        observed_durations = observed_durations[0]

        n_pos_phases = np.sum(index_pos_phase_durations)
        sum_durations = np.sum(
            np.arange(np.size(index_pos_phase_durations)) *
            index_pos_phase_durations)

        p_hat = n_pos_phases / sum_durations

        bootstrap_p_hat = np.empty((n_bootstrap,), dtype='f8')
        original_sample = np.repeat(observed_durations, observed_counts)

        for i in range(n_bootstrap):
            bootstrap_sample = rng.choice(original_sample, n_pos_phases,
                                          replace=True)
            bootstrap_counts = np.bincount(bootstrap_sample)
            bootstrap_sum = np.sum(
                np.arange(np.size(bootstrap_counts)) *
                bootstrap_counts)
            bootstrap_p_hat[i] = n_pos_phases / bootstrap_sum

        bootstrap_p_hat = np.sort(bootstrap_p_hat)
        lower_index = int(np.floor(0.5 * alpha * n_bootstrap))
        upper_index = int(np.ceil((1 - 0.5 * alpha) * n_bootstrap))

        geometric_fits[index]['mle'] = p_hat
        geometric_fits[index]['bootstrap_se'] = np.std(bootstrap_p_hat, ddof=1)
        geometric_fits[index]['bootstrap_conf_int'] = (
            bootstrap_p_hat[lower_index], bootstrap_p_hat[upper_index])
        geometric_fits[index]['n_bootstrap'] = n_bootstrap
        geometric_fits[index]['alpha'] = alpha

    return geometric_fits


def plot_empirical_distribution(pos_phase_duration_counts, n_cols=2,
                                is_daily=False):
    index_names = sorted([f for f in pos_phase_duration_counts])
    n_indices = len(index_names)

    n_rows = int(np.ceil(n_indices / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=True,
                             figsize=(8, 11))
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4, wspace=0.3)

    row_index = 0
    col_index = 0
    for i, index in enumerate(index_names):
        index_duration_counts = pos_phase_duration_counts[index]
        n_pos_phases = np.sum(index_duration_counts)

        possible_durations = np.arange(np.size(index_duration_counts))
        empirical_dist = np.cumsum(index_duration_counts) / n_pos_phases

        axes[row_index, col_index].step(
            possible_durations, empirical_dist)

        axes[row_index, col_index].set_ylabel('Empirical CDF')

        if is_daily:
            axes[row_index, col_index].set_xlabel('Duration (days)')
        else:
            axes[row_index, col_index].set_xlabel('Duration (months)')

        axes[row_index, col_index].set_title(index)

        col_index += 1
        if col_index == n_cols:
            row_index += 1
            col_index = 0

    plt.show()


def plot_pos_phase_duration_counts(pos_phase_duration_counts,
                                   pos_phase_geometric_fits=None,
                                   is_daily=False, n_cols=2,
                                   density=True):
    index_names = sorted([f for f in pos_phase_duration_counts])
    n_indices = len(index_names)

    n_rows = int(np.ceil(n_indices / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=True,
                             figsize=(8, 11))
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4, wspace=0.3)

    row_index = 0
    col_index = 0
    for i, index in enumerate(index_names):
        index_duration_counts = pos_phase_duration_counts[index]

        possible_durations = np.arange(np.size(index_duration_counts))
        observed_durations = np.nonzero(index_duration_counts)
        observed_values = index_duration_counts[observed_durations]
        observed_durations = observed_durations[0]

        n_pos_phases = np.sum(index_duration_counts)

        if density:
            observed_values = observed_values / n_pos_phases

        axes[row_index, col_index].plot(
            observed_durations, observed_values, 'ko',
            label='observed')

        if pos_phase_geometric_fits is not None:
            index_p_hat = pos_phase_geometric_fits[index]['mle']
            index_p_hat_ci = pos_phase_geometric_fits[index]['bootstrap_conf_int']
            fitted_values = geometric_pmf(possible_durations[1:], index_p_hat)

            lower_values = geometric_pmf(possible_durations[1:],
                                         index_p_hat_ci[0])
            upper_values = geometric_pmf(possible_durations[1:],
                                         index_p_hat_ci[1])

            lower_ci = np.fmin(lower_values, upper_values)
            upper_ci = np.fmax(lower_values, upper_values)

            if not density:
                fitted_values = fitted_values * n_pos_phases
                lower_ci = lower_ci * n_pos_phases
                upper_ci = upper_ci * n_pos_phases

            axes[row_index, col_index].semilogy(
                possible_durations[1:], fitted_values,
                color='orange', lw=1.05,
                label='p_hat = {:.3e}'.format(index_p_hat))
            axes[row_index, col_index].semilogy(
                possible_durations[1:], lower_ci,
                color='orange', lw=1, alpha=0.5)
            axes[row_index, col_index].semilogy(
                possible_durations[1:], upper_ci,
                color='orange', lw=1, alpha=0.5)

            axes[row_index, col_index].fill_between(
                possible_durations[1:],
                fitted_values, upper_ci,
                color='orange', alpha=0.5)
            axes[row_index, col_index].fill_between(
                possible_durations[1:],
                lower_ci, fitted_values,
                color='orange', alpha=0.5)


        axes[row_index, col_index].legend()

        if density:
            axes[row_index, col_index].set_ylabel('Proportion')
        else:
            axes[row_index, col_index].set_ylabel('Frequency')

        if is_daily:
            axes[row_index, col_index].set_xlabel('Duration (days)')
        else:
            axes[row_index, col_index].set_xlabel('Duration (months)')
        axes[row_index, col_index].set_title(index)

        col_index += 1
        if col_index == n_cols:
            row_index += 1
            col_index = 0

    plt.show()


def print_separator(length=70):
    print(length * '-')


def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description='Print statistics for index signs time-series')

    parser.add_argument('input_csv_file', help='input CSV file')
    parser.add_argument('--show-plots', dest='show_plots', action='store_true',
                        help='show plots')

    args = parser.parse_args()

    return args.input_csv_file, args.show_plots


def main():
    n_bootstrap = 1000
    alpha = 0.05

    random_state = np.random.RandomState(0)

    input_csv_file, show_plots = parse_cmd_line_args()

    times, values, is_daily_data = read_data(input_csv_file)

    index_names = [f for f in values]

    print_separator()
    print('Input file: %s' % input_csv_file)
    print('Indices: %r' % index_names)
    print('Start time: %s' % np.min(times).isoformat())
    print('End time: %s' % np.max(times).isoformat())
    if is_daily_data:
        print('Frequency: daily')
    else:
        print('Frequency: monthly')
    print_separator()

    phase_fractions = calculate_phase_fractions(values)
    print('Positive phase fractions:')
    for f in phase_fractions:
        print('\t%s: %.3e' % (f, phase_fractions[f]))
    print_separator()

    pos_phase_duration_counts = get_positive_phase_duration_frequencies(values)
    pos_phase_summary_stats = get_positive_phase_duration_summary(
        pos_phase_duration_counts)
    pos_phase_duration_fits = get_positive_phase_duration_geometric_fits(
        pos_phase_duration_counts, n_bootstrap=n_bootstrap, alpha=alpha,
        random_state=random_state)

    print('Positive phase duration summary statistics:')
    print('| Index | Minimum | LQ | Mean | Median | Mode | UQ | Maximum |')
    print('--------------------------------------------------------------')
    for index in pos_phase_summary_stats:
        print('| {} | {:d} | {:d} | {:.2f} | {:d} | {:d} | {:d} | {:d} |'.format(
            index, pos_phase_summary_stats[index]['minimum'],
            pos_phase_summary_stats[index]['lower_quartile'],
            pos_phase_summary_stats[index]['mean'],
            pos_phase_summary_stats[index]['median'],
            pos_phase_summary_stats[index]['mode'],
            pos_phase_summary_stats[index]['upper_quartile'],
            pos_phase_summary_stats[index]['maximum']))
    print_separator()
    print('Positive phase duration geometric fits:')
    print('| Index | MLE p_hat | SE(p_hat) | Lower | Upper | n_bootstrap | alpha |')
    for index in pos_phase_duration_fits:
        print('| {} | {:.5e} | {:.5e} | {:.5e} | {:.5e} | {:d} | {:.3e} |'.format(
            index, pos_phase_duration_fits[index]['mle'],
            pos_phase_duration_fits[index]['bootstrap_se'],
            pos_phase_duration_fits[index]['bootstrap_conf_int'][0],
            pos_phase_duration_fits[index]['bootstrap_conf_int'][1],
            n_bootstrap, alpha))

    if show_plots:
        plot_empirical_distribution(pos_phase_duration_counts,
                                    is_daily=is_daily_data)
        plot_pos_phase_duration_counts(pos_phase_duration_counts,
                                       pos_phase_duration_fits,
                                       is_daily=is_daily_data)


if __name__ == '__main__':
    main()
