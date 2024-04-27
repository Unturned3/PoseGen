
import numpy as np

def multimodal(func, locs, scales, weights=None):

    assert len(locs) == len(scales)

    if weights is None:
        weights = [1 / len(locs)] * len(locs)
    else:
        assert len(locs) == len(weights)

    distributions = [func(loc=l, scale=s) for l, s in zip(locs, scales)]
    choices = np.random.choice(len(distributions), size=1, p=weights)
    samples = np.array([distributions[i].rvs() for i in choices])
    return samples[0]
