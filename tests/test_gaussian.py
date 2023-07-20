import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from composite_tests.distributions.gaussian import (
    Gaussian,
    GaussianFixedScaleMMDEstimator,
)
from composite_tests.kernels import GaussianKernel


class TestGaussianFixedScaleMMDEstimator:
    def test__call__result_is_reasonable(self):
        rng = PRNGKey(seed=109875117)

        true_scale = 0.8
        kernel = GaussianKernel(l=1.0)
        estimator = GaussianFixedScaleMMDEstimator(kernel, true_scale)

        for true_loc in [0.1, -2.1, 0.8]:
            for repeat in range(3):
                rng, rng_input = jax.random.split(rng)
                ys = Gaussian(true_loc, true_scale).sample(rng_input, n=200)

                rng, rng_input = jax.random.split(rng)
                estimated = estimator(rng_input, ys)

                error = jnp.abs(estimated - true_loc)
                assert error < 0.2, f"Est {estimated}, true {true_loc}, rep {repeat}"
