from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap
from jax.random import PRNGKey

from composite_tests.distributions.gaussian import Gaussian
from composite_tests.rejection_sampler import sample


def test__sample__gaussian__has_correct_mean_and_variance():
    rng = PRNGKey(seed=12345)
    true_mean = 0.8
    true_std = 1.0
    n = 40000

    proposal = Gaussian(loc=0.0, scale=3.0)
    target_log_pdf = partial(jax.scipy.stats.norm.logpdf, loc=true_mean, scale=true_std)

    rng, rng_input = jax.random.split(rng)
    xs, _ = sample(rng_input, proposal, target_log_pdf, n)

    empirical_mean = xs.mean()
    empirical_std = xs.std()

    assert xs.shape == (n, 1)
    assert jnp.abs(true_mean - empirical_mean) < 0.01
    assert jnp.abs(true_std - empirical_std) < 0.01


def test__sample__gaussians_in_vmap__have_correct_means_and_variances():
    rng = PRNGKey(seed=12345)
    true_mean = 0.8
    true_std = 1.0
    n = 40000
    repeats = 10

    proposal = Gaussian(loc=0.0, scale=3.0)
    target_log_pdf = partial(jax.scipy.stats.norm.logpdf, loc=true_mean, scale=true_std)

    rng, *rng_inputs = jax.random.split(rng, num=repeats + 1)
    xs, _ = vmap(sample, in_axes=(0, None, None, None))(
        jnp.stack(rng_inputs), proposal, target_log_pdf, n
    )

    empirical_mean = xs.mean(1)
    empirical_std = xs.std(1)

    assert xs.shape == (repeats, n, 1)
    assert jnp.all(jnp.abs(true_mean - empirical_mean) < 0.01)
    assert jnp.all(jnp.abs(true_std - empirical_std) < 0.01)
