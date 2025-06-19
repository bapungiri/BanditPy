import numpy as np


def generate_probs_2arm(p=None, N=1000, frac_impurity=0.16):
    """Generate probabilities for a 2-arm bandit task from a given set of probabilities. Where for,
        unstructured environments: the sum of probabilities is not equal to 1 and includes a minority of strucutured probabilities.

        structured environments: the sum of probabilities is equal to 1 and includes a minority of unstructured probabilities.


    Note: Equal probabilities are not allowed, i.e., p1 != p2.


    Parameters
    ----------
    p : _type_
        _description_
    N : int, optional
        Number of probabilities to generate, by default 1000.
    frac_impurity : float
        Fraction of probabilities that are impure (i.e., fraction of unstructured in structured and vice versa).

    Returns
    -------
    tuple
        Tuple containing two numpy arrays:
        - pu_new: Unstructured probabilities (shape: N x 2)
        - ps_new: Structured probabilities (shape: N x 2)
    """
    if p is None:
        p = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])

    assert frac_impurity < 0.5, "Fraction of impure probabilities must be less than 0.5"

    rng = np.random.default_rng()

    n_impure = int(N * frac_impurity)  # number of divergent probs
    n_pure = N - n_impure  # number of structured probs

    # structured probabilities
    p1s = np.random.choice(p, size=n_pure, replace=True)
    p2s = np.round(1 - p1s, 1)
    ps = np.array([p1s, p2s]).T

    # stuctured impurities (i.e unstructured probabilities)
    p1su = np.random.choice(p, size=n_impure, replace=True)
    p2su = np.array(
        [np.random.choice(np.setdiff1d(p, [val, np.round(1 - val, 1)])) for val in p1su]
    )
    psu = np.array([p1su, p2su]).T

    ps_new = np.concatenate((ps, psu), axis=0)
    rng.shuffle(ps_new)

    # unstructured probabilities
    p1u = np.random.choice(p, size=n_pure, replace=True)
    p2u = np.array(
        [np.random.choice(np.setdiff1d(p, [val, np.round(1 - val, 1)])) for val in p1u]
    )
    pu = np.array([p1u, p2u]).T

    # unstructured impurities (i.e structured probabilities)
    p1us = np.random.choice(p, size=n_impure, replace=True)
    p2us = np.round(1 - p1us, 1)
    pus = np.array([p1us, p2us]).T

    pu_new = np.concatenate((pu, pus), axis=0)
    rng.shuffle(pu_new)

    # Count how many impure probabilities
    ps_sum = ps_new.sum(axis=1)
    pu_sum = pu_new.sum(axis=1)
    frac_u_in_s = np.sum(ps_sum != 1) / N  # unstructured in structured
    frac_s_in_u = np.sum(pu_sum == 1) / N  # structured in unstructured
    frac_equal_s = np.sum(ps_new[:, 0] == ps_new[:, 1]) / N
    frac_equal_u = np.sum(pu_new[:, 0] == pu_new[:, 1]) / N

    assert frac_equal_s == 0, "There are equal probabilities in structured"
    assert frac_equal_u == 0, "There are equal probabilities in unstructured"

    assert frac_u_in_s == frac_impurity, "frac_impurity violated in structured"
    assert frac_s_in_u == frac_impurity, "frac_impurity violated in unstructured"

    return pu_new, ps_new
