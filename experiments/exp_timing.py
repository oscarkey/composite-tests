import time

import jax
from jax.numpy import ndarray
from jax.random import KeyArray, PRNGKey
from tqdm import tqdm

from composite_tests.bootstrapped_tests import (
    parametric_bootstrap_test,
    wild_bootstrap_test,
)
from composite_tests.distributions.gaussian import Gaussian
from composite_tests.kernels import GaussianKernel
from composite_tests.ksd import KSDAnalyticEstimator, KSDStatistic


def main() -> None:
    rng = PRNGKey(seed=8230482)

    null_dist = Gaussian
    kernel = GaussianKernel(l=1.0)
    estimator = KSDAnalyticEstimator(kernel, null_dist)
    statistic = KSDStatistic(kernel, null_dist)

    def run_parametric(rng: KeyArray) -> None:
        rng_input1, rng_input2 = jax.random.split(rng)
        parametric_bootstrap_test(
            rng_input1,
            _sample_observations(rng_input2),
            estimator,
            null_dist,
            statistic,
            n_bootstrap_samples=300,
            save_null_distribution=False,
        )

    def run_wild(rng: KeyArray) -> None:
        rng_input1, rng_input2 = jax.random.split(rng)
        wild_bootstrap_test(
            rng_input1,
            _sample_observations(rng_input2),
            estimator,
            statistic,
            n_bootstrap_samples=300,
            save_null_distribution=False,
        )

    # We do some tests first to warm up the jit and anything else.
    for _ in range(3):
        rng, rng_input = jax.random.split(rng)
        run_wild(rng_input)
        rng, rng_input = jax.random.split(rng)
        run_parametric(rng_input)

    # We don't need to worry about lazy evaluation because the test functions call
    # .item() on the test result, which blocks.
    n_repeats = 10
    start_time = time.time()
    for _ in tqdm(range(n_repeats)):
        rng, rng_input = jax.random.split(rng)
        run_wild(rng_input)
    millis_per_wild = 1000 * (time.time() - start_time) / n_repeats
    start_time = time.time()
    for _ in tqdm(range(n_repeats)):
        rng, rng_input = jax.random.split(rng)
        run_parametric(rng_input)
    millis_per_parametric = 1000 * (time.time() - start_time) / n_repeats
    print(f"Wild took {millis_per_wild:.2f}ms per test")
    print(f"Para took {millis_per_parametric:.2f}ms per test")


OBSERVATION_DIST = Gaussian(loc=0.5, scale=1.0)


def _sample_observations(rng: KeyArray) -> ndarray:
    return OBSERVATION_DIST.sample(rng, n=100)


if __name__ == "__main__":
    main()
