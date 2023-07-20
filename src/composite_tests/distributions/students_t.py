from dataclasses import dataclass
from typing import NamedTuple, Union

import jax.numpy as jnp
import jax.random
import scipy.stats
from jax import Array
from jax.random import KeyArray

from composite_tests.distributions import SampleableDist
from composite_tests.extra_types import Scalar


class UnivariateT(SampleableDist[Scalar, Scalar]):
    def __init__(self, df: Union[float, Scalar]) -> None:
        self.df = jnp.array(df)

    def get_params(self) -> Scalar:
        return self.df

    @staticmethod
    def sample_with_params(rng: KeyArray, params: Scalar, n: int) -> Scalar:
        return jax.random.t(rng, df=params, shape=(n, 1))


class MVTParams(NamedTuple):
    df: Scalar
    mean: Array
    cov: Array


class MultivariateT(SampleableDist[MVTParams, Scalar]):
    """Multivariate Student's t distribution.

    Uses Scipy implementation, which is slow compared to a JAX native one.

    :param cov: the desired covariance of the samples, the distribution scale parameter
                will be set appropriately to achieve this
    """

    def __init__(self, df: float, mean: Array, cov: Array) -> None:
        self.params = MVTParams(jnp.array(df), mean, cov)

    def get_params(self) -> MVTParams:
        return self.params

    @staticmethod
    def sample_with_params(rng: KeyArray, params: MVTParams, n: int) -> Scalar:
        scale = (params.cov * (params.df - 2.0)) / params.df
        dist = scipy.stats.multivariate_t(
            params.mean, scale, params.df, seed=int(rng[0])
        )
        return dist.rvs(size=n)
