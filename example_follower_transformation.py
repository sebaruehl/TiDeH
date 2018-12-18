"""
Example of predicting future retweet activity using optimized Python implementations and different transformation of
follower counts.

This code is developed by Sebastian Rühl.
"""
from tideh import load_events_vec
from tideh import estimate_parameters_optimized
from tideh import predict_optimized
from tideh import follower_tanh
from tideh import follower_random_network

filename = 'data/example/sample_file.txt'
obs_time = 48       # observation time of 2 days
pred_time = 168     # predict for one week

# the number of retweets is not necessary for the further steps
# make sure that all times are loaded in the correct time unit (hours)
# here it is important that there is one nd-array for event times and one for the follower counts
(_, start_time), (event_times, follower) = load_events_vec(filename)

# additional parameters passed to infectious rate function
add_params = {'t0': start_time, 'bounds': [(-1, 0.5), (1, 20.)]}


# standard
params, err, _ = estimate_parameters_optimized(event_times=event_times, follower=follower, obs_time=obs_time,
                                               **add_params)

print("Estimated parameters standard method:")
print("p0:   %.3f, r0:   %.3f, phi0: %.3f, tm:   %.3f, error (estimated to fitted): %.2f " %
      (params[0], params[1], params[2], params[3], err * 100))

# predict future retweets
_, _, pred_error = predict_optimized(event_times=event_times, follower=follower, obs_time=obs_time,
                                     pred_time=pred_time, p_max=None, params=params, **add_params)

print("Prediction error standard method: %.0f\n" % pred_error)

# tanh transformation
A = 500000
follower_new_tanh = follower_tanh(follower, A)

params, err, _ = estimate_parameters_optimized(event_times=event_times, follower=follower_new_tanh, obs_time=obs_time,
                                               **add_params)

print("Estimated parameters tanh method:")
print("p0:   %.3f, r0:   %.3f, phi0: %.3f, tm:   %.3f, error (estimated to fitted): %.2f " %
      (params[0], params[1], params[2], params[3], err * 100))

# predict future retweets
_, _, pred_error = predict_optimized(event_times=event_times, follower=follower_new_tanh, obs_time=obs_time,
                                     pred_time=pred_time, p_max=None, params=params, **add_params)

print("Prediction error tanh method: %.0f\n" % pred_error)


# random network transformation
p = 0.0001
follower_new_rn = follower_random_network(follower, p)

params, err, _ = estimate_parameters_optimized(event_times=event_times, follower=follower_new_rn, obs_time=obs_time,
                                               **add_params)

print("Estimated parameters random network method:")
print("p0:   %.3f, r0:   %.3f, phi0: %.3f, tm:   %.3f, error (estimated to fitted): %.2f " %
      (params[0], params[1], params[2], params[3], err * 100))

# predict future retweets
_, _, pred_error = predict_optimized(event_times=event_times, follower=follower_new_rn, obs_time=obs_time,
                                     pred_time=pred_time, p_max=None, params=params, **add_params)

print("Prediction error random network method: %.0f" % pred_error)
