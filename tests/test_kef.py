import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate as integrate
from jax.random import PRNGKey

from composite_tests.distributions.kef import (
    compute_js_and_sqrt_factorials,
    kernel_exp_family,
)


class TestKernelExpFamily:
    def test__compute_norm_constant__makes_density_integrate_to_one(self):
        rng = PRNGKey(seed=987654321)
        p = 5
        rng, rng_input = jax.random.split(rng)
        theta = jax.random.normal(rng_input, shape=(p,))
        distribution = kernel_exp_family(p, l=1.0)(theta)

        rng, rng_input = jax.random.split(rng)
        norm_constant = distribution.compute_norm_constant(rng_input, n_samples=10000)

        norm_density = lambda x: distribution.unnorm_prob(jnp.array(x)) / norm_constant
        integrated, _ = integrate.quad(norm_density, -np.inf, np.inf)
        error = np.abs(integrated - 1.0)
        assert error < 0.01


def test__compute_js_and_sqrt_factorials__returns_correct_sqrt_factorials():
    _, actual = compute_js_and_sqrt_factorials(p=4)
    expected = jnp.array(
        [jnp.sqrt(1), jnp.sqrt(1 * 2), jnp.sqrt(1 * 2 * 3), jnp.sqrt(1 * 2 * 3 * 4)]
    )
    assert jnp.allclose(actual, expected)
