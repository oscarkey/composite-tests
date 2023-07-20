import jax.numpy as jnp

import composite_tests.kernels as kernels
from composite_tests.kernels import GaussianKernel, SumKernel


class TestSumKernel:
    def test__eq_hash__kernels_equal__returns_True(self) -> None:
        k1 = SumKernel([GaussianKernel(l=1.0), GaussianKernel(l=0.5)])
        k2 = SumKernel([GaussianKernel(l=1.0), GaussianKernel(l=0.5)])
        assert hash(k1) == hash(k2)
        assert k1 == k2

    def test__eq_hash__kernels_not_equal__returns_False(self) -> None:
        k1 = SumKernel([GaussianKernel(l=1.0), GaussianKernel(l=0.4)])
        k2 = SumKernel([GaussianKernel(l=1.0), GaussianKernel(l=0.5)])
        assert hash(k1) != hash(k2)
        assert k1 != k2


def test__gram__standard_kernel__has_correct_dimensions():
    xs = jnp.full((14, 1), 1.0)
    ys = jnp.full((21, 1), 2.0)
    kernel = GaussianKernel(l=1.0)

    gram = kernels.gram(xs, ys, kernel)

    assert gram.shape == (14, 21)
