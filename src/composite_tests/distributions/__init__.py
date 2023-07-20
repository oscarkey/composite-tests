"""Distributions where multiple instances can share the same sample/pdf functions."""
from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

from jax import Array
from jax.random import KeyArray

X = TypeVar("X")
P = TypeVar("P")


class Distribution(ABC, Generic[P]):
    @abstractmethod
    def get_params(self) -> P:
        pass


class SampleableDist(Distribution[P], ABC, Generic[P, X]):
    def sample(self, rng: KeyArray, n: int) -> X:
        return self.sample_with_params(rng, self.get_params(), n)

    @staticmethod
    @abstractmethod
    def sample_with_params(rng: KeyArray, params: P, n: int) -> X:
        pass


class UnnormalizedDist(Distribution[P], ABC, Generic[P]):
    def score(self, x: Array) -> Array:
        return self.score_with_params(self.get_params(), x)

    @staticmethod
    @abstractmethod
    def score_with_params(params: P, x: Array) -> Array:
        pass

    def unnorm_log_prob(self, x: Array) -> Array:
        return self.unnorm_log_prob_with_params(self.get_params(), x)

    @classmethod
    @abstractmethod
    def unnorm_log_prob_with_params(cls, params: P, x: Array) -> Array:
        pass


class NormalizedDist(UnnormalizedDist[P], ABC, Generic[P]):
    def log_prob(self, x: Array) -> Array:
        return self.log_prob_with_params(self.get_params(), x)

    @staticmethod
    @abstractmethod
    def log_prob_with_params(params: P, x: Array) -> Array:
        pass

    @classmethod
    def unnorm_log_prob_with_params(cls, params: P, x: Array) -> Array:
        return cls.log_prob_with_params(params, x)


class ExpFamilyDist(Protocol):
    @staticmethod
    @abstractmethod
    def natural_parameter(params: Array) -> Array:
        """param space -> R^k"""
        pass

    @staticmethod
    @abstractmethod
    def natural_parameter_inverse(eta_val: Array) -> Array:
        """R^k -> param space"""
        pass

    @staticmethod
    @abstractmethod
    def sufficient_statistic(x: Array) -> Array:
        """R^d -> R^k"""
        pass

    @staticmethod
    @abstractmethod
    def b(x: Array) -> Array:
        """R^d -> R"""
        # TODO: Use proper name for b.
        pass


class SampleableAndNormalizedDist(
    SampleableDist[P, X], NormalizedDist[P], ABC, Generic[P, X]
):
    pass


class SampleableAndUnnormalizedDist(
    SampleableDist[P, X], UnnormalizedDist[P], ABC, Generic[P, X]
):
    pass
