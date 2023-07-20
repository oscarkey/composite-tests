import jax
import jax.numpy as jnp
import jax.scipy
import matplotlib.pyplot as plt
from jax.random import PRNGKey

from composite_tests.distributions.gaussian import (
    Gaussian,
    gaussian_fixed_scale,
    mvn_fixed_cov,
)
from composite_tests.distributions.kef import kernel_exp_family
from composite_tests.kernels import GaussianKernel
from composite_tests.ksd import KSDAnalyticEstimator, KSDStatistic


class TestKSDAnalyticEstimator:
    def test__gaussian_fixed_scale__returns_close_to_true_mean(self):
        rng = PRNGKey(seed=109875117)
        kernel = GaussianKernel(l=1.0)

        for scale in [1.0, 1.5]:
            dist = gaussian_fixed_scale(scale)
            estimator = KSDAnalyticEstimator(kernel, dist)
            for true_loc in [0.0, 2.5, 8.0]:
                rng, rng_input = jax.random.split(rng)
                ys = dist(true_loc).sample(rng_input, n=2000)

                rng, rng_input = jax.random.split(rng)
                estimated_loc = estimator(rng_input, ys)[0].item()

                error = abs(estimated_loc - true_loc)
                assert error < 0.07, f"est={estimated_loc}, true={true_loc}"

    def test__gaussian_both_params__returns_close_to_true_params(self):
        rng = PRNGKey(seed=109875117)
        kernel = GaussianKernel(l=1.0)
        for true_scale in [0.6, 1.0, 1.2]:
            for true_loc in [2.5, 0.0, 8.0]:
                dist = Gaussian(true_loc, true_scale)
                estimator = KSDAnalyticEstimator(kernel, dist)

                rng, rng_input = jax.random.split(rng)
                ys = dist.sample(rng_input, n=2000)

                rng, rng_input = jax.random.split(rng)
                estimated_params = estimator(rng_input, ys)
                est_loc, est_scale = estimated_params[0], estimated_params[1]

                loc_error = abs(est_loc - true_loc)
                scale_error = abs(est_scale - true_scale)
                assert loc_error < 0.06
                assert scale_error < 0.06

    def test__kef__returns_close_to_true_values(self):
        rng = PRNGKey(seed=109875117)

        p = 2
        q0_std = 1.0
        true_l = 1.0
        rng, rng_input = jax.random.split(rng)
        true_theta = jax.random.normal(rng_input, shape=(p,))
        dist = kernel_exp_family(p, true_l, q0_std=q0_std)

        rng, rng_input = jax.random.split(rng)
        ys = dist(true_theta).sample(rng_input, n=5000)

        kernel = GaussianKernel(l=0.5)
        estimator = KSDAnalyticEstimator(kernel, dist)

        rng, rng_input = jax.random.split(rng)
        theta_est = estimator(rng_input, ys)

        errors = jnp.abs(theta_est - true_theta)
        assert jnp.all(errors < 0.05)

    def test__mvn_mean__independent_dimensions__returns_close_to_true_value(self):
        rng = PRNGKey(seed=123456789)

        true_loc = jnp.array([0.2, 0.2])
        true_cov = 0.6 * jnp.eye(2)
        dist = mvn_fixed_cov(true_cov)

        rng, rng_input = jax.random.split(rng)
        ys = dist(true_loc).sample(rng_input, n=5000)

        kernel = GaussianKernel(l=1.5)
        estimator = KSDAnalyticEstimator(kernel, dist)

        rng, rng_input = jax.random.split(rng)
        theta_est = estimator(rng_input, ys)

        errors = jnp.abs(theta_est - true_loc)
        assert jnp.all(errors < 0.01), f"errors = {errors}"

    def test__mvn_mean__dependent_dimensions__returns_close_to_true_value(self):
        rng = PRNGKey(seed=123456769)

        true_loc = jnp.array([1.2, 0.8])
        true_cov = jnp.array([[1.6, 0.2], [0.2, 1.7]])
        dist = mvn_fixed_cov(true_cov)

        rng, rng_input = jax.random.split(rng)
        ys = dist(true_loc).sample(rng_input, n=5000)

        kernel = GaussianKernel(l=1.5)
        estimator = KSDAnalyticEstimator(kernel, dist)

        rng, rng_input = jax.random.split(rng)
        theta_est = estimator(rng_input, ys)

        errors = jnp.abs(theta_est - true_loc)
        assert jnp.all(errors < 0.01), f"errors = {errors}"


class TestKSDStatistic:
    def test__gaussian_fixed_scale__null_samples_closer_to_each_other_than_alt_samples(
        self,
    ):
        rng = PRNGKey(seed=29794824)

        null_family = gaussian_fixed_scale(scale=1.0)
        null_dist = null_family(loc=0.0)
        alt_1_dist = gaussian_fixed_scale(scale=1.5)(loc=0.0)
        alt_2_dist = gaussian_fixed_scale(scale=2.0)(loc=1.0)
        statistic = KSDStatistic(GaussianKernel(l=1.0), null_family)

        rng, rng_input = jax.random.split(rng)
        null_stat = statistic(
            rng_input, null_dist.get_params(), null_dist.sample(rng_input, n=200)
        )
        rng, rng_input = jax.random.split(rng)
        alt_1_stat = statistic(
            rng_input, null_dist.get_params(), alt_1_dist.sample(rng_input, n=200)
        )
        rng, rng_input = jax.random.split(rng)
        alt_2_stat = statistic(
            rng_input, null_dist.get_params(), alt_2_dist.sample(rng_input, n=200)
        )

        assert null_stat < alt_1_stat
        assert null_stat < alt_2_stat

    def test__null_case__gaussian__statistic_approaches_zero(self):
        rng = PRNGKey(seed=29794824)
        dist = Gaussian(loc=10.0, scale=0.5)

        rng, rng_input = jax.random.split(rng)
        stat = KSDStatistic(GaussianKernel(l=1.0), null=Gaussian)

        ns = [10, 100, 500, 1000]
        # We sample the ys upfront and just use a subset each time to reduce variance.
        ys = dist.sample(rng_input, max(ns))
        last_stat = jnp.array(100.0)
        for i, n in enumerate(ns):
            rng, rng_input = jax.random.split(rng)
            s = stat(rng_input, theta_hat=jnp.array([10.0, 0.5]), ys=ys[:n])
            assert s <= last_stat, f"failed at {ns[i]} < {ns[i-1]}"
            last_stat = s

    def test__null_case__kef__statistic_approaches_zero(self):
        rng = PRNGKey(seed=25794824)
        dist_family = kernel_exp_family(p=5, l=2.0)
        rng, rng_input = jax.random.split(rng)
        theta = jax.random.uniform(rng_input, shape=(5,), minval=-1.0, maxval=1.0)
        dist = dist_family(theta)

        rng, rng_input = jax.random.split(rng)
        stat = KSDStatistic(GaussianKernel(l=1.0), null=dist_family)

        ns = [10, 100, 500, 1000]
        # We sample the ys upfront and just use a subset each time to reduce variance.
        ys = dist.sample(rng_input, max(ns))
        last_stat = jnp.array(100.0)
        for i, n in enumerate(ns):
            rng, rng_input = jax.random.split(rng)
            s = stat(rng_input, theta_hat=theta, ys=ys[:n])
            assert s <= last_stat, f"failed at {ns[i]} < {ns[i-1]}"
            last_stat = s
