import numpy as np
import matplotlib.pyplot as plt

def generate_default_times(intensity = 0.2, censure_intensity = 0.08, n_obs=100):
    times = np.random.exponential(1/intensity, n_obs)
    censures = np.random.exponential(1/censure_intensity, n_obs)
    return [min(times[k], censures[k]) for k in range(n_obs)], [1 if times[k] < censures[k] else 0for k in range(n_obs)]

def kaplan_meier_estimator(defaults, censures, bins=100):
    times = np.linspace(min(defaults), max(defaults), bins)
    ordered_times = sorted(defaults)
    survival_estimation = {x : 0 for x in ordered_times}
    failures_tk = []
    at_risks = []
    for t in ordered_times:
        failures_tk += [np.sum([censures[k] for k in range(len(defaults)) if defaults[k] == t])]
        at_risks += [np.sum([1 for k in range(len(defaults)) if defaults[k] > t])]
    for t in ordered_times:
        survival_estimation[t] = np.prod([1-(failures_tk[k]/at_risks[k]) for k in range(ordered_times.index(t))])
    return survival_estimation

if __name__ == "__main__":
    intensity = 0.2
    np.random.seed(1)
    defaults, censures = generate_default_times(intensity=intensity)
    print("true value : ", intensity)
    print("MLE estimator without censorship : ", len(defaults)/np.sum(defaults))
    print("MLE estimator with censorship : ", np.sum(censures)/np.sum(defaults))
    survival_estimators = kaplan_meier_estimator(defaults, censures)
    plt.figure()
    plt.plot(list(survival_estimators.keys()), list(survival_estimators.values()), label="KM")
    plt.plot(list(survival_estimators.keys()), [np.exp(-(np.sum(censures)/np.sum(defaults))*list(survival_estimators.keys())[k]) for k in range(len(survival_estimators.keys()))], label="MLE estimation with censure")
    plt.plot(list(survival_estimators.keys()), [np.exp(-(len(defaults)/np.sum(defaults))*list(survival_estimators.keys())[k]) for k in range(len(survival_estimators.keys()))], label="MLE estimation without censure")
    plt.plot(list(survival_estimators.keys()), [np.exp(-0.2*list(survival_estimators.keys())[k]) for k in range(len(survival_estimators.keys()))], label="real function")
    plt.legend()
    plt.title("Survival function estimations")
    plt.show()
