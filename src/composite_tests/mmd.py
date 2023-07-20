from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

from jax import Array, jit
from jax.random import KeyArray

import composite_tests.kernels as kernels
from composite_tests.bootstrapped_tests import WildTestStatistic
from composite_tests.distributions import SampleableDist
from composite_tests.kernels import Kernel

P = TypeVar("P")


@dataclass(frozen=True)
class MMDStatistic(WildTestStatistic, Generic[P]):
    kernel: Kernel
    null_dist: type[SampleableDist[P, Array]]

    def __call__(self, rng: KeyArray, theta_hat: P, ys: Array) -> Array:
        xs = self.null_dist.sample_with_params(rng, theta_hat, ys.shape[0])
        return v_stat(xs, ys, self.kernel)

    def h_gram(
        self,
        rng: KeyArray,
        theta_hat: Array,
        ys: Array,
    ) -> Array:
        xs = self.null_dist.sample_with_params(rng, theta_hat, ys.shape[0])
        return h_gram(xs, ys, self.kernel)

    @property
    def name(self) -> str:
        return f"mmd"


@partial(jit, static_argnames=("kernel"))
def v_stat(xs: Array, ys: Array, kernel: Kernel) -> Array:
    return h_gram(xs, ys, kernel).mean()


@partial(jit, static_argnames=("kernel"))
def h_gram(xs: Array, ys: Array, kernel: Kernel) -> Array:
    assert len(xs) == len(ys)
    K_xx = kernels.gram(xs, xs, kernel)
    K_yy = kernels.gram(ys, ys, kernel)
    K_xy = kernels.gram(xs, ys, kernel)
    return K_xx + K_yy - K_xy - K_xy.T
