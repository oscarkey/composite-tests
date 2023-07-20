from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import jax
import jax.numpy as jnp
from jax import Array, vmap

from composite_tests.extra_types import Scalar

KernelLike = Callable[[Array, Array], Array]


class Kernel(ABC):
    @abstractmethod
    def __call__(self, x1: Array, x2: Array) -> Scalar:
        pass


# It's important that frozen,eq=True so dataclasses implements a hash function, which
# avoids JAX recompiling unecessarily if the kernel is the same.
# TODO: Use Chex dataclasses to avoid recompiling when only a kernel parameter (e.g. the
# lengthscale) changes.
@dataclass(frozen=True, eq=True)
class GaussianKernel(Kernel):
    l: float

    def __call__(self, x1: Array, x2: Array) -> Scalar:
        return jnp.exp(-((x1 - x2) ** 2).sum() / (2 * self.l**2))


@dataclass(frozen=True, eq=True)
class IMQKernel(Kernel):
    l: float
    gamma: float = 0.5

    def __post_init__(self) -> None:
        assert self.l >= 0.0
        assert 0.0 <= self.gamma and self.gamma <= 1.0

    def __call__(self, x1: Array, x2: Array) -> Scalar:
        return (1 + ((x1 - x2) ** 2).sum() / (2 * self.l**2)) ** -self.gamma


class SumKernel(Kernel):
    def __init__(self, kernels: Iterable[Kernel]) -> None:
        self.kernels = tuple(kernels)

    def __call__(self, x1: Array, x2: Array) -> Array:
        return jnp.array([k(x1, x2) for k in self.kernels]).sum()

    def __hash__(self) -> int:
        return hash(self.kernels)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, SumKernel):
            return False
        if len(self.kernels) != len(o.kernels):
            return False
        return all([k1 == k2 for (k1, k2) in zip(self.kernels, o.kernels)])


def gram(x1: Array, x2: Array, kernel: KernelLike) -> Array:
    """Computes the gram matrix for a kernel

    :param x1: [n x d]
    :param x2: [n x d]
    :return: [n x n]
    """
    assert x1.ndim == 2
    assert x2.ndim == 2

    return vmap(lambda x: vmap(lambda y: kernel(x, y))(x2))(x1)


@jax.jit
def median_heuristic(x1: Array, x2: Optional[Array] = None) -> Array:
    if x2 is None:
        xs = x1
    else:
        xs = jnp.concatenate([x1, x2])

    distances = vmap(lambda xa: vmap(lambda xb: ((xa - xb) ** 2).sum() / 2)(xs))(xs)
    return jnp.sqrt(jnp.median(distances))
