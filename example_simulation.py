"""
This code generates a retweet dataset based on Time-Dependent Hawkes process (TiDeH).
Then, the model parameters are estimated from the dataset.

Inputs are
1) Model parameters of TiDeH (p_0, r_0, phi_0, t_m).
2) Observation time (= obs_time).

Outputs are
1) Estimate of model parameters of TiDeH (p_0, r_0, phi_0, t_m).

This code is developed by Sylvain Gauthier and Sebastian Rühl under the supervision of Ryota Kobayashi.
"""
from tideh.simulate import simulate_time_rescaling
from tideh.functions import infectious_rate_tweets
from tideh import load_events_vec
from tideh import estimate_parameters_optimized


# load pool of follower counts used for simulation from file
file_path = 'data/example/sample_file.txt'
_, (_, follower_pool) = load_events_vec(file_path)

runtime = 72  # simulate for 3 days
# parameters of infectious rate
p0 = 0.001
r0 = 0.424
phi0 = 0.125
taum = 2.

# simulate
event_times, follower = simulate_time_rescaling(runtime=runtime,
                                                p=lambda t: infectious_rate_tweets(t, p0, r0, phi0, taum),
                                                follower_pool=follower_pool[1:], int_fol_cnt=follower_pool[0])

# estimate original infectious rate parameters
add_params = {'bounds': [(-1, 0.5), (1, 20.)]}
params, err, _ = estimate_parameters_optimized(event_times, follower, runtime, **add_params)

print("Number of simulated events: %i" % len(event_times))
print("Estimated parameters are (actual value):")
print("p0:   %.5f (%0.3f)" % (params[0], p0))
print("r0:   %.5f (%0.3f)" % (params[1], r0))
print("phi0: %.5f (%0.3f)" % (params[2], phi0))
print("tm:   %.5f (%0.3f)" % (params[3], taum))
print("Average %% error (estimated to fitted): %.2f" % (err * 100))




