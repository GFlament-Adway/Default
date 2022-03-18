import numpy as np

def generate_default_times(intensity = 0.2, censure_intensity = 0.08, n_obs=100):
    times = np.random.exponential(1/intensity, n_obs)
    censures = np.random.exponential(1/censure_intensity, n_obs)
    return [min(times[k], censures[k]) for k in range(n_obs)], [1 if times[k] < censures[k] else 0for k in range(n_obs)]

    