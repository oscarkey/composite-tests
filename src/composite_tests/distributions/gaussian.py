from dataclasses import dataclass
from functools import partial
from typing import Union

import jax.numpy as jnp
import jax.random
import optax
from jax import Array, grad, jit
from jax.random import KeyArray
from jax.scipy.stats import multivariate_normal, norm

import composite_tests.mmd as mmd
from composite_tests.distributions import ExpFamilyDist, SampleableAndNormalizedDist
from composite_tests.estimators import Estimator
from composite_tests.extra_types import Scalar
from composite_tests.jax_utils import to_scalar
from composite_tests.kernels import Kernel
from composite_tests.optimizers import run_optimizer


class Gaussian(SampleableAndNormalizedDist[Array, Array], ExpFamilyDist):
    def __init__(self, loc: Union[Scalar, float], scale: Union[Scalar, float]) -> None:
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)

    def prob(self, x: Array) -> Array:
        return norm.pdf(x, loc=self.loc, scale=self.scale)

    @staticmethod
    def sample_with_params(rng: KeyArray, params: Array, n: int) -> Array:
        return _sample(rng, loc=params[0], scale=params[1], n=n)

    @staticmethod
    def score_with_params(params: Array, x: Scalar) -> Array:
        return _score(x, loc=params[0], scale=params[1])

    @staticmethod
    def log_prob_with_params(params: Array, x: Scalar) -> Array:
        return norm.logpdf(x, params[0], params[1])

    def get_params(self) -> Array:
        return jnp.array([self.loc, self.scale])

    @staticmethod
    def natural_parameter(params: Array) -> Array:
        loc, scale = params[0], params[1]
        return jnp.array([loc / scale, -1 / (2 * scale)])

    @staticmethod
    def natural_parameter_inverse(eta_val: Array) -> Array:
        loc = -0.5 * eta_val[0] / eta_val[1]
        scale = jnp.sqrt(1 / (-2 * eta_val[1]))
        return jnp.array([loc, scale])

    @staticmethod
    def sufficient_statistic(x: Array) -> Array:
        return jnp.concatenate([x, x**2], axis=0)

    @staticmethod
    def b(x: Array) -> Array:
        # Note that the KSD only depends on db/dx. For Gaussians, b does not depend on
        # x, thus db/dx = 0. Thus, to simplify things, we just return zero here.
        return jnp.zeros(shape=())


def gaussian_fixed_scale(scale: Union[float, Scalar]):
    s = jnp.array(scale)

    class GaussianFixedScale(SampleableAndNormalizedDist[Array, Array], ExpFamilyDist):
        def __init__(self, loc: Scalar) -> None:
            self.loc = jnp.array([loc])

        def prob(self, x: Array) -> Array:
            return norm.pdf(x, loc=self.loc, scale=s)

        @staticmethod
        def sample_with_params(rng: KeyArray, params: Array, n: int) -> Array:
            return _sample(rng, loc=params[0], scale=s, n=n)

        @staticmethod
        def score_with_params(params: Array, x: Scalar) -> Array:
            return _score(x, loc=params[0], scale=s)

        @staticmethod
        def log_prob_with_params(params: Array, x: Scalar) -> Array:
            return norm.logpdf(x, params[0], s)

        def get_params(self) -> Array:
            return self.loc

        @staticmethod
        def natural_parameter(params: Array) -> Array:
            mean = params[0]
            return mean / s**2

        @staticmethod
        def natural_parameter_inverse(eta_val: Array) -> Array:
            mean = -0.5 * eta_val[0] / eta_val[1]
            std = jnp.sqrt(1 / (-2 * eta_val[1]))
            return jnp.array([mean, std])

        @staticmethod
        def sufficient_statistic(x: Array) -> Array:
            return jnp.concatenate([x, x**2], axis=0)

        @staticmethod
        def b(x: Array) -> Array:
            # Note that the KSD only depends on db/dx. For Gaussians, b does not depend on
            # x, thus db/dx = 0. Thus, to simplify things, we just return zero here.
            return jnp.zeros(shape=())

    return GaussianFixedScale


@partial(jit, static_argnames=("n"))
def _sample(rng: KeyArray, loc: Scalar, scale: Scalar, n: int) -> Array:
    return loc + scale * jax.random.normal(rng, shape=(n, 1))


def _score(x: Array, loc: Scalar, scale: Scalar) -> Array:
    return grad(to_scalar(norm.logpdf), argnums=0)(x.reshape(x.shape[0]), loc, scale)


def mvn_fixed_cov(cov: Array):
    assert cov.ndim == 2
    assert cov.shape[0] == cov.shape[1]
    d = cov.shape[0]
    cov_inv = jnp.linalg.inv(cov)

    @dataclass
    class MVNFixedCovar(SampleableAndNormalizedDist[Array, Array]):
        loc: Array

        def __post_init__(self) -> None:
            self.loc.shape[0] == d

        @staticmethod
        def sample_with_params(rng: KeyArray, params: Array, n: int) -> Array:
            loc = params
            return jax.random.multivariate_normal(rng, loc, cov, shape=(n,))

        @staticmethod
        def score_with_params(params: Array, x: Array) -> Array:
            loc = params
            # The type of logpdf doesn't indicate that if you pass in an array you get
            # an array back. We need an array back so we can reshape it, so the line
            # below is a hack to fix the type.
            array_logpdf = lambda x: jnp.array(multivariate_normal.logpdf(x, loc, cov))
            return grad(to_scalar(array_logpdf), argnums=0)(
                x.reshape(x.shape[0]), loc, cov
            )

        @staticmethod
        def log_prob_with_params(params: Array, x: Array) -> Array:
            loc = params
            return jnp.array(multivariate_normal.logpdf(x, loc, cov))

        def get_params(self) -> Array:
            return self.loc

        @staticmethod
        def natural_parameter(params: Array) -> Array:
            return params

        @staticmethod
        def natural_parameter_inverse(eta_val: Array) -> Array:
            return eta_val

        @staticmethod
        def sufficient_statistic(x: Array) -> Array:
            return (x @ cov_inv).T

        @staticmethod
        def b(x: Array) -> Array:
            return -0.5 * x @ cov_inv @ x.T

    return MVNFixedCovar


@dataclass(frozen=True, eq=True)
class GaussianMMDEstimator(Estimator):
    """Estimates the mean and variance of a Gaussian using SGD on the MMD v-stat."""

    kernel: Kernel
    iterations: int = 40
    learning_rate: float = 0.5

    def __call__(self, rng: KeyArray, ys: Array) -> Array:
        return _optimize_gaussian(
            rng,
            self.kernel,
            ys,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            m=ys.shape[0],
        )

    @property
    def name(self) -> str:
        return f"mmd"


@partial(jit, static_argnames=("kernel", "iterations", "learning_rate", "m"))
def _optimize_gaussian(
    rng: KeyArray,
    kernel: Kernel,
    ys: Array,
    iterations: int,
    learning_rate: float,
    m: int,
) -> Array:
    def loss(rng2: KeyArray, params: Array) -> Array:
        xs = Gaussian.sample_with_params(rng2, params, n=m)
        return mmd.v_stat(xs, ys, kernel)

    return run_optimizer(
        rng,
        optimizer=optax.sgd(learning_rate),
        loss=loss,
        theta_init=jnp.array([0.01, 1.0]),
        iterations=iterations,
    )


@dataclass(frozen=True, eq=True)
class GaussianFixedScaleMMDEstimator(Estimator):
    """Estimates the mean of a Gaussian with known std using SGD on the MMD v-stat."""

    kernel: Kernel
    scale: Array
    iterations: int = 30
    learning_rate: float = 0.5

    def __call__(self, rng: KeyArray, ys: Array) -> Array:
        return _optimize_gaussian_mean_fixed_scale(
            rng,
            self.kernel,
            self.scale,
            ys,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            m=ys.shape[0],
        )

    @property
    def name(self) -> str:
        return f"mmd"


@partial(jit, static_argnames=("kernel", "iterations", "learning_rate", "m"))
def _optimize_gaussian_mean_fixed_scale(
    rng: KeyArray,
    kernel: Kernel,
    scale: Array,
    ys: Array,
    iterations: int,
    learning_rate: float,
    m: int,
) -> Array:
    dist_family = gaussian_fixed_scale(scale)

    def loss(rng2: KeyArray, mean: Array) -> Array:
        xs = dist_family.sample_with_params(rng2, mean, n=m)
        return mmd.v_stat(xs, ys, kernel)

    return run_optimizer(
        rng,
        optimizer=optax.sgd(learning_rate),
        loss=loss,
        theta_init=jnp.array([0.01]),
        iterations=iterations,
    )
