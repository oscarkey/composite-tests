from functools import partial
from typing import Generic, Protocol, Type, TypeVar

import jax.numpy as jnp
from jax import Array, grad, jacfwd, jacrev, jit
from jax.numpy import atleast_2d
from jax.random import KeyArray

from composite_tests.bootstrapped_tests import WildTestStatistic
from composite_tests.distributions import ExpFamilyDist, UnnormalizedDist
from composite_tests.estimators import Estimator
from composite_tests.extra_types import Scalar
from composite_tests.kernels import Kernel, gram


class KSDAnalyticEstimator(Estimator[Array]):
    def __init__(self, kernel: Kernel, dist_family: ExpFamilyDist) -> None:
        self.kernel = kernel
        self.dist_family = dist_family

    def __call__(self, rng: KeyArray, ys: Array) -> Array:
        return ksd_estimator_exp_family(self.kernel, self.dist_family, ys)

    @property
    def name(self) -> str:
        return "ksd"


P = TypeVar("P")


class KSDStatistic(WildTestStatistic[P], Generic[P]):
    def __init__(self, kernel: Kernel, null: Type[UnnormalizedDist]) -> None:
        self.kernel = kernel
        self.null = null

    def __call__(self, rng: KeyArray, theta_hat: P, ys: Array) -> Array:
        return v_stat(self.kernel, self.null.score_with_params, theta_hat, ys)

    def h_gram(self, rng: KeyArray, theta_hat: P, ys: Array) -> Array:
        return h_gram(self.kernel, self.null.score_with_params, theta_hat, ys)

    @property
    def name(self) -> str:
        return f"ksd"


class ScoreFunc(Protocol):
    @staticmethod
    def __call__(params: Array, x: Array) -> Array:
        pass


class NaturalParameterInverse(Protocol):
    @staticmethod
    def __call__(eta_val: Array) -> Array:
        pass


class SufficientStatistic(Protocol):
    @staticmethod
    def __call__(x: Array) -> Array:
        pass


class B(Protocol):
    @staticmethod
    def __call__(x: Array) -> Array:
        pass


@partial(jit, static_argnames=("kernel", "score"))
def v_stat(kernel: Kernel, score: ScoreFunc, params: Array, ys: Array) -> Scalar:
    return h_gram(kernel, score, params, ys).mean()


@partial(jit, static_argnames=("kernel", "score"))
def h_gram(kernel: Kernel, score: ScoreFunc, params: Array, ys: Array) -> Array:
    def h(y1: Array, y2: Array) -> Array:
        term1 = kernel(y1, y2) * score(params, y1) @ score(params, y2)
        term2 = score(params, y1) @ grad(kernel, argnums=1)(y1, y2)
        term3 = score(params, y2) @ grad(kernel, argnums=0)(y1, y2)
        term4 = jacfwd(jacrev(kernel, 0), 1)(y1, y2).trace()
        return term1 + term2 + term3 + term4

    return gram(ys, ys, h)


@partial(jit, static_argnames=("kernel", "dist_family"))
def ksd_estimator_exp_family(
    kernel: Kernel,
    dist_family: Type[ExpFamilyDist],
    ys: Array,
) -> Array:
    eta_inv = dist_family.natural_parameter_inverse
    t = dist_family.sufficient_statistic
    b = dist_family.b

    def Lambda(y1: Array, y2: Array) -> Array:
        return kernel(y1, y2) * atleast_2d(jacfwd(t)(y1)) @ atleast_2d(jacfwd(t)(y2)).T

    def nu(y1: Array, y2: Array) -> Array:
        term1 = kernel(y1, y2) * jacfwd(b)(y1) @ jacfwd(t)(y2).T
        term2 = atleast_2d(jacfwd(t)(y1)) @ jacfwd(kernel, argnums=1)(y1, y2)
        term3 = kernel(y1, y2) * jacfwd(b)(y2) @ jacfwd(t)(y1).T
        term4 = atleast_2d(jacfwd(t)(y2)) @ jacfwd(kernel, argnums=0)(y1, y2)
        return term1 + term2 + term3 + term4

    # big_lambda gram matrix has dimensions [n x n x k x k].
    # k is the dimension of eta.
    big_lambda_n = gram(ys, ys, Lambda).mean(0).mean(0)
    # Add a little bit to the diagonal to improve stability. Otherwise the matrix
    # inverse can explode.
    big_lambda_n = big_lambda_n + jnp.eye(big_lambda_n.shape[0]) * 1e-4

    # nu gram matrix has dimensions [n x n x k]
    nu_n = gram(ys, ys, nu)
    nu_n = nu_n.mean(0).mean(0)

    # We use linalg.solve rather than linalg.inv because it is more stable as it avoids
    # explicitly computing the matrix inverse.
    eta_estimate = jnp.linalg.solve(big_lambda_n, -0.5 * nu_n)

    theta_estimate = eta_inv(eta_estimate)
    return theta_estimate
