import jax
import jax.numpy as jnp
import optax
from jax import Array
from jax.random import KeyArray, PRNGKey

import composite_tests.mmd as mmd
from composite_tests.kernels import GaussianKernel
from composite_tests.optimizers import random_restart_optimizer


def test__random_restart_optimizer__result_is_reasonable():
    rng = PRNGKey(seed=14324)

    # Use a Gaussian mixture as the model.
    def sample_gaussian_mixture(rng: KeyArray, theta: Array) -> Array:
        rng, rng_input = jax.random.split(rng)
        means = jax.random.choice(rng_input, theta, shape=(40, 1))
        rng, rng_input = jax.random.split(rng)
        return means + jax.random.normal(rng_input, shape=means.shape)

    true_theta = jnp.array([-0.5, 0.5])
    rng, rng_input = jax.random.split(rng)
    ys = sample_gaussian_mixture(rng_input, true_theta)

    kernel = GaussianKernel(1.0)

    def loss(rng: KeyArray, theta: Array) -> Array:
        xs = sample_gaussian_mixture(rng, theta)
        return mmd.v_stat(xs, ys, kernel)

    optimizer = optax.sgd(learning_rate=0.02)
    sample_init_theta = lambda rng: jax.random.uniform(
        rng, shape=(2,), minval=jnp.array([-1.0, -1.0]), maxval=jnp.array([1.0, 1.0])
    )

    rng, rng_input = jax.random.split(rng)
    theta_hat = random_restart_optimizer(
        rng_input,
        optimizer,
        loss,
        sample_init_theta,
        iterations=0,
        n_initial_locations=200,
        n_optimized_locations=5,
    )

    error = jnp.abs(theta_hat - true_theta)
    assert jnp.max(error).item() < 0.1
