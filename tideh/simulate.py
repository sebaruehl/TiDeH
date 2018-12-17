# Author:   Sylvain Gauthier
# Author:   Sebastian Rühl
#
# For license information, see LICENSE.txt

"""
Implements functions for simulating time dependent Hawkes process following the Twitter model.

Provides different functions for simulating counts of follower, in the simplest approach they are extracted from a file
containing real observation data.

References
----------
.. Kobayashi, R. and Lambiotte, R., 2016, March. TiDeH: Time-Dependent Hawkes Process for Predicting Retweet Dynamics.
   In ICWSM (pp. 191-200).
"""

from . import functions
from math import *
import numpy as np
import numpy.random as rand


def rand_followers(scale_factor=100):
    """
    Generates random amount followers following a exponential distribution.

    :param scale_factor: mean of followers
    :return: randomly generated follower count
    """
    return round(-scale_factor * log(rand.uniform()))


def rand_followers_extended(initial, scale_factor=100, split=0.02):
    """
    Generation of followers where sometimes the follower count can be very big (relative to the given initial value).

    :param initial: some initial follower value, should be big
    :param scale_factor: mean of followers
    :param split: percentage for when the follower should be generated very big
    :return: randomly generated follower count
    """
    rn = rand.uniform()
    if rn > split:
        return round(-scale_factor * log(rand.uniform()))
    else:
        return round(rand.uniform(0.05 * initial, 0.6 * initial))


def solve_integral(ti, X, kernel, p, events, dt, Tmax):
    """
    Helper function for simulation using time rescaling.
    """
    partial_sum = 0
    last_partial_sum = 0
    t = ti
    lambda_0 = p(t) * sum([fol_count * kernel(t - event_time) for event_time, fol_count in events])
    lambda_1 = None
    while partial_sum < X:
        t += dt
        lambda_1 = p(t) * sum([fol_count * kernel(t - event_time) for event_time, fol_count in events])
        partial_sum += dt * (lambda_0 + lambda_1) / 2

        if partial_sum < X:
            lambda_0 = lambda_1
            last_partial_sum = partial_sum
        if t > Tmax:
            return -1

    dlam = (lambda_1 - lambda_0) / dt
    du = X - last_partial_sum
    s = (sqrt(lambda_0 * lambda_0 + 2 * dlam * du) - lambda_0) / dlam
    return t - dt + s


def simulate_time_rescaling(runtime, kernel=functions.kernel_zhao, p=functions.infectious_rate_tweets, dt=0.01,
                            follower_pool=None, int_fol_cnt=10000, follower_mean=200, split=0.015):
    """
    Simulates time dependent Hawkes process using time rescaling.

    Follower counts can be taken from a pool passed to the function or generated.

    :param runtime: time to simulate (in hours)
    :param kernel: kernel function
    :param p: infectious rate function
    :param dt: integral evaluation interval size
    :param follower_pool: follower counts used for simulation, makes last 2 parameters void
    :param int_fol_cnt: initial follower value
    :param follower_mean: mean of generated followers
    :param split: percentage for when the follower should be generated very big
    :return: list of event tuples
    """
    events = [(0, int_fol_cnt)]
    ti = 0
    print_cnt = 0

    while 0 <= ti < runtime and len(events) < 4500:
        X = rand.exponential()
        tj = solve_integral(ti, X, kernel, p, events, dt, runtime)
        if follower_pool is not None:
            fol = rand.choice(follower_pool)
        else:
            fol = rand_followers_extended(int_fol_cnt, follower_mean, split)
        if tj > 0:
            events.append((tj, fol))
        ti = tj
        if print_cnt % 100 == 0:
            print("Simulating [%f%%]..." % (ti / runtime * 100), flush=True)
        print_cnt += 1

    print("\nOver %d events generated" % len(events))

    return events


def simulate_hawkes_time_increment(runtime, dt=0.01, kernel=functions.kernel_zhao_vec,
                                   p=functions.infectious_rate_tweets, follower_pool=None, int_fol_cnt=10000,
                                   follower_mean=200, split=0.02):
    """
    Simulates time dependent Hawkes process using time increments.
    Keeps track of all interesting intermediate values.

    Follower counts can be taken from a pool passed to the function or generated.

    Very slow for big run times.

    :param runtime: time to simulate (in hours)
    :param dt: time steps
    :param kernel: kernel function
    :param p: infectious rate function
    :param follower_pool: follower counts used for simulation, makes last 3 parameters void
    :param int_fol_cnt: initial follower count for follower generation
    :param follower_mean: mean value of follower for follower generation
    :param split: split parameter for follower generation
    :return: 3-tuple, holding list tuple of event_imes and follower nd-arrays, list of intensity for every interval,
    list of memory effect for every interval
    """
    event_times = np.array([0])
    follower = np.array([int_fol_cnt])

    lambda_t = [p(0) * int_fol_cnt * kernel(0)]
    memory_effect_t = [int_fol_cnt * kernel(0)]

    for cur_interval in np.arange(0, runtime - dt, dt):

        memory_effect = sum(follower * kernel(cur_interval - event_times))
        llambda = p(cur_interval) * memory_effect * dt  # intensity for current interval

        lambda_t.append(llambda)
        memory_effect_t.append(memory_effect)

        if rand.uniform() < llambda:  # event occurred
            if follower_pool is not None:
                fol = rand.choice(follower_pool)
            else:
                fol = rand_followers_extended(int_fol_cnt, follower_mean, split)
            event_times = np.append(event_times, cur_interval)
            follower = np.append(follower, fol)

    return (event_times, follower), lambda_t, memory_effect_t
