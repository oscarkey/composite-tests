from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jax import Array
from jax.random import KeyArray

P = TypeVar("P")


class Estimator(ABC, Generic[P]):
    @abstractmethod
    def __call__(self, rng: KeyArray, ys: Array) -> P:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


@dataclass
class TrueEstimator(Estimator[P]):
    """An estimator that just returns a given value, useful for debugging.

    You can use this to turn a composite test into a non-composite test, by creating an
    estimator that just returns the true value of the parameter.
    """

    theta: P

    def __call__(self, rng: KeyArray, ys: Array) -> P:
        return self.theta

    @property
    def name(self) -> str:
        return "true"
