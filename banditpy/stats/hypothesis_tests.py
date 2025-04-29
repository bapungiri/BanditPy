import numpy as np
from scipy.stats import bootstrap


def monte_carlo_bootstrap_test(
    x, y, statistic, n_samples=10000, two_sided=True, paired=False, random_state=None
):

    # Observed difference
    observed_diff = statistic(x) - statistic(y)

    # Define statistic function for bootstrap — must take a tuple of arrays
    def stat_diff(x, y):
        return statistic(x) - statistic(y)

    # Perform bootstrap
    result = bootstrap(
        data=(x, y),
        statistic=stat_diff,
        vectorized=False,
        paired=paired,
        n_resamples=n_samples,
        # method="basic",
        random_state=random_state,
    )

    # The bootstrap distribution is available via confidence_interval — we compute p-value manually:
    resampled_diffs = result.bootstrap_distribution

    if two_sided:
        p_value = np.mean(np.abs(resampled_diffs) >= np.abs(observed_diff))
    else:
        p_value = np.mean(resampled_diffs >= observed_diff)

    return observed_diff, p_value
