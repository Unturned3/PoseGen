
import numpy as np
from scipy.stats import norm

def bimodal_normal(weights, means, std_devs):
    """
    Generate a bimodal distribution by mixing two normal distributions.

    Parameters:
        n (int): Number of points to generate.
        weights (list of float): Weights for each sub-distribution.
        means (list of float): Means for each sub-distribution.
        std_devs (list of float): Standard deviations for each sub-distribution.

    Returns:
        np.array: Random numbers from the specified bimodal distribution.
    """
    distributions = [norm(loc=mean, scale=std) for mean, std in zip(means, std_devs)]
    choices = np.random.choice(len(distributions), size=1, p=weights)
    samples = np.array([distributions[i].rvs() for i in choices])
    return samples[0]
