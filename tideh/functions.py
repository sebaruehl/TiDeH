# Author:   Sylvain Gauthier
# Author:   Sebastian Rühl
#
# For license information, see LICENSE.txt

"""
Implements basic mathematical expression of different functions used for estimation, prediction and simulation.
This includes the memory kernel, integral of memory kernel and the infectious rate.

Provides implementations using native Python and optimized versions using nd-arrays and vectorization.

References
----------
.. Kobayashi, R. and Lambiotte, R., 2016, March. TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics.
   In ICWSM (pp. 191-200).
.. Zhao, Q., Erdogdu, M.A., He, H.Y., Rajaraman, A. and Leskovec, J., 2015, August. Seismic: A self-exciting point
   process model for predicting tweet popularity. In Proceedings of the 21th ACM SIGKDD International Conference on
   Knowledge Discovery and Data Mining (pp. 1513-1522). ACM.
"""

from math import *
import numpy as np


def kernel_zhao(s, s0=0.08333, theta=0.242):
    """
    Calculates Zhao kernel for given value.

    :param s: time point to evaluate
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: value at time point s
    """
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)  # normalization constant
    if s >= 0:
        if s <= s0:
            return c0
        else:
            return c0 * (s / s0) ** (-(1. + theta))
    else:
        return 0


def kernel_zhao_vec(s, s0=0.08333, theta=0.242):
    """
    Calculates Zhao kernel for given value.
    Optimized using nd-arrays and vectorization.

    :param s: time points to evaluate, should be a nd-array
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: values at given time points
    """
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)  # normalization constant
    res = np.copy(s)
    res[s < 0] = 0
    res[(s <= s0) & (s >= 0)] = c0
    res[s > s0] = c0 * (res[s > s0] / s0) ** (-(1. + theta))
    return res


def kernel_primitive_zhao(x, s0=0.08333, theta=0.242):
    """
    Calculates the primitive of the Zhao kernel for given values.

    :param x: point to evaluate
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: primitive evaluated at x
    """
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)
    if x < 0:
        return 0
    elif x <= s0:
        return c0 * x
    else:
        return c0 * (s0 + (s0 * (1 - (x / s0) ** -theta)) / theta)


def kernel_primitive_zhao_vec(x, s0=0.08333, theta=0.242):
    """
    Calculates the primitive of the Zhao kernel for given values.
    Optimized using nd-arrays and vectorization.

    :param x: points to evaluate, should be a nd-array
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :param c0: normalization constant
    :return: primitives evaluated at given points
    """
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)
    res = np.copy(x)
    res[x < 0] = 0
    res[(x <= s0) & (x >= 0)] = c0 * res[(x <= s0) & (x >= 0)]
    res[x > s0] = c0 * (s0 + (s0 * (1 - (res[x > s0] / s0) ** -theta)) / theta)
    return res


def integral_zhao(x1, x2, s0=0.08333, theta=0.242):
    """
    Calculates definite integral of Zhao function.

    :param x1: start
    :param x2: end
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: integral of Zhao function
    """
    return kernel_primitive_zhao(x2, s0, theta) - kernel_primitive_zhao(x1, s0, theta)


def integral_zhao_vec(x1, x2, s0=0.08333, theta=0.242):
    """
    Calculates definite integral of Zhao function.
    Optimized using nd-arrays and vectorization.

    x1 and x2 should be nd-arrays of same size.

    :param x1: start values
    :param x2: end values
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: integrals of Zhao function
    """
    return kernel_primitive_zhao_vec(x2, s0, theta) - kernel_primitive_zhao_vec(x1, s0, theta)


def infectious_rate_tweets(t, p0=0.001, r0=0.424, phi0=0.125, taum=2., t0=0, tm=24, bounds=None):
    """
    Alternative form of infectious rate from paper. Supports bounds for r0 and taum. Bounds should be passed as an array
    in the form of [(lower r0, lower taum), (upper r0, upper taum)].
    Converted to hours.

    :param t: point to evaluate function at (in hours)
    :param p0: base rate
    :param r0: amplitude
    :param phi0: shift (in days)
    :param taum: decay/freshness (in days)
    :param t0: start time of observation (in hours)
    :param tm: cyclic property (after what time a full circle passed, in hours)
    :param bounds: bounds for r0 and taum
    :return: infectiousness for time t
    """
    if bounds is not None:
        if not (bounds[0][0] < r0 < bounds[1][0]):
            r0 = max(bounds[0][0], bounds[1][0] * sigmoid(taum / bounds[1][0]))
        if not (bounds[0][1] < taum < bounds[1][1]):
            taum = max(bounds[0][1], bounds[1][1] * sigmoid(taum / bounds[1][1]))
    return p0 * (1. - r0 * sin((48 / tm) * pi * ((t + t0) / 24 + phi0))) * exp(-t / (24 * taum))


def infectious_rate_tweets_vec(t, p0=0.001, r0=0.424, phi0=0.125, taum=2., t0=0, tm=24., bounds=None):
    """
    Alternative form of infectious rate from paper. Supports bounds for r0 and taum. Bound should be passed as an array
    in the form of [(lower r0, lower taum), (upper r0, upper taum)].
    Converted to hours.
    Vectorized version.

    :param t: points to evaluate function at, should be a nd-array (in hours)
    :param p0: base rate
    :param r0: amplitude
    :param phi0: shift (in days)
    :param taum: decay/freshness (in days)
    :param t0: start time of observation (in hours)
    :param tm: cyclic property (after what time a full circle passed, in hours)
    :param bounds: bounds for r0 and taum
    :return: infectiousness for given t
    """
    if bounds is not None:
        if not (bounds[0][0] < r0 < bounds[1][0]):
            r0 = max(bounds[0][0], bounds[1][0] * sigmoid(taum / bounds[1][0]))
        if not (bounds[0][1] < taum < bounds[1][1]):
            taum = max(bounds[0][1], bounds[1][1] * sigmoid(taum / bounds[1][1]))
    return p0 * (1. - r0 * np.sin((48. / tm) * np.pi * ((t + t0) / 24. + phi0))) * np.exp(-t / (24. * taum))


def infectious_rate_dv_p0(t, r0=0.424, phi0=0.125, taum=2., t0=0, tm=24.):
    """
    Derivation of infectious rate after p0.

    Required for direct maximum likelihood estimation.

    :param t: points to evaluate function at, shoult be nd-arrays (in hours)
    :param r0:  amplitude
    :param phi0:  shift (in days)
    :param taum:  decay/freshness (in days)
    :param t0: start time of observation (in hours)
    :param tm: cyclic property (after what a fill circle passed, on hours)
    :return: infectious rate derived after p0
    """
    return (1. - r0 * np.sin((48. / tm) * np.pi * ((t + t0) / 24. + phi0))) * np.exp(-t / (24. * taum))


def sigmoid(x):
    """
    Calculates sigmoid function for value x.
    """
    return 1 / (1 + exp(-x))


def get_event_count(event_times, start, end):
    """
    Count of events in given interval.

    :param event_times: nd-array of event times
    :param start: interval start
    :param end: interval end
    :return: count of events in interval
    """
    mask = (event_times > start) & (event_times <= end)
    return event_times[mask].size


def prediction_error_absolute(event_times, intensity, window_size, obs_time, pred_time, dt):
    """
    Calculates absolute prediction error.

    :param event_times: event times
    :param intensity: predicted intensity
    :param window_size: prediction window size
    :param obs_time: observation time
    :param pred_time: prediction time
    :param dt: interval width for numerical integral calculation used for intensity prediction
    :return: absolute prediction error
    """
    events_time_pred = event_times[event_times >= obs_time]
    win_int = int(window_size / dt)
    tp = np.arange(obs_time, pred_time, window_size)
    err = 0
    for i, t_cur in enumerate(tp):
        t_end = t_cur + window_size
        if t_end > pred_time:
            break
        count_current = get_event_count(events_time_pred, t_cur, t_end)
        pred_count = dt * intensity[(i * win_int):((i + 1) * win_int)].sum()
        err += abs(count_current - pred_count)

    return err


def prediction_error_normed(event_times, intensity, window_size, obs_time, pred_time, dt):
    """
    Calculates normed prediction error.

    :param event_times: event times
    :param intensity: predicted intensity
    :param window_size: prediction window size
    :param obs_time: observation time
    :param pred_time: prediction time
    :param dt: interval width for numerical integral calculation used for intensity prediction
    :return: normed prediction error
    """
    err_abs = prediction_error_absolute(event_times, intensity, window_size, obs_time, pred_time, dt)

    events_time_pred = event_times[event_times >= obs_time]
    total_count_real = get_event_count(events_time_pred, obs_time, pred_time)

    if total_count_real == 0 and err_abs == 0:
        err_normed = 0
    elif total_count_real == 0:
        err_normed = 10  # if there are no events in prediction period set error to very high value
    else:
        err_normed = err_abs / total_count_real

    return err_normed


def prediction_error_relative(event_times, intensity, window_size, obs_time, pred_time, dt):
    """
    Calculates median relative running error.

    :param event_times: event times
    :param intensity: predicted intensity
    :param window_size: prediction window size
    :param obs_time: observation time
    :param pred_time: prediction time
    :param dt: interval width for numerical integral calculation used for intensity prediction
    :return: normed prediction error
    """
    events_time_pred = event_times[event_times >= obs_time]
    events_in_obs_time = get_event_count(events_time_pred, 0, obs_time)

    win_int = int(window_size / dt)
    tp = np.arange(obs_time, pred_time, window_size)
    cnt_total_real = events_in_obs_time
    cnt_total_pred = events_in_obs_time
    err_rel_running = []
    for i, t_cur in enumerate(tp):
        t_end = t_cur + window_size
        if t_end > pred_time:
            break
        count_current = get_event_count(events_time_pred, t_cur, t_end)  # real event count in interval
        pred_count = dt * intensity[(i * win_int):((i + 1) * win_int)].sum()  # predicted event count in interval

        cnt_total_real += count_current
        cnt_total_pred += pred_count
        rel_tmp = 1 - (np.minimum(cnt_total_real, cnt_total_pred) / np.maximum(cnt_total_real, cnt_total_pred))
        err_rel_running.append(rel_tmp)

    return np.median(err_rel_running)
