import numpy as np
import torch


def uniform_weights(n_sample):
    """Return uniform weights (almost for debug).

    EXAMPLE
    -------
    >>> weights = uniform_weights(3)
    >>> print(weights)
    <BLANKLINE>
     0.3333
     0.3333
     0.3333
    [torch.FloatTensor of size 3]
    <BLANKLINE>

    :return:
    """
    weights = torch.ones(n_sample)
    return weights / weights.sum()


def raml_weights(rewards, proposed_weights, tau):
    """Return exp-scaled weights.

    EXAMPLE
    -------
    >>> rewards = np.array([0, -1, -1])
    >>> proposed_weights = np.array([1., 1., 1.])
    >>> weights = raml_weights(rewards, proposed_weights, 1.0)
    >>> weights.sum()
    1.0

    :param rewards:
    :param proposed_weights:
    :return:
    """

    exp_rewards = np.exp(rewards / tau)
    weights = exp_rewards * proposed_weights
    weights = weights / weights.sum()

    return torch.from_numpy(weights)
