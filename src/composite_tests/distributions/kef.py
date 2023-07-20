import math
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import Array, grad, jit, vmap
from jax.random import KeyArray

import composite_tests.rejection_sampler as rejection_sampler
from composite_tests.distributions import ExpFamilyDist, SampleableAndUnnormalizedDist
from composite_tests.distributions.gaussian import Gaussian
from composite_tests.extra_types import Scalar
from composite_tests.jax_utils import to_scalar


def kernel_exp_family(p: int, l: float, q0_mean: float = 0.0, q0_std: float = 1.0):
    q0 = Gaussian(q0_mean, q0_std)
    # q0 should work well as the proposal because it controls the behaviour of
    # the tails of the distribution.
    proposal = Gaussian(q0_mean, scale=q0_std + 0.5)

    class KernelExpFamily(SampleableAndUnnormalizedDist[Array, Scalar], ExpFamilyDist):
        def __init__(self, theta: Array) -> None:
            self.theta = theta

        def get_params(self) -> Array:
            return self.theta

        @staticmethod
        def sample_with_params(rng: KeyArray, params: Array, n: int) -> Array:
            x, acceptance_rate = rejection_sampler.sample(
                rng,
                proposal,
                lambda x: KernelExpFamily.unnorm_log_prob_with_params(params, x),
                n,
            )
            return x

        @staticmethod
        @jit
        def score_with_params(params: Array, x: Array) -> Array:
            return grad(
                to_scalar(KernelExpFamily.unnorm_log_prob_with_params), argnums=1
            )(params, x)

        @staticmethod
        @jit
        def unnorm_log_prob_with_params(params: Array, x: Array) -> Array:
            assert (x.ndim == 1 and x.shape[0] == 1) or x.shape == ()
            assert params.ndim == 1

            return q0.log_prob(x) + jnp.dot(params, phi(x, l, p))

        @staticmethod
        def natural_parameter(params: Array) -> Array:
            return params

        @staticmethod
        def natural_parameter_inverse(eta_val: Array) -> Array:
            return eta_val

        @staticmethod
        def sufficient_statistic(x: Array) -> Array:
            return phi(x, l, p)

        @staticmethod
        def b(x: Array) -> Array:
            return q0.log_prob(x).reshape(())

        @staticmethod
        @jit
        def unnorm_prob_with_params(params: Array, x: Array) -> Array:
            assert (x.ndim == 1 and x.shape[0] == 1) or x.shape == ()
            assert params.ndim == 1
            return q0.prob(x) * jnp.exp(jnp.dot(params, phi(x, l, p)))

        def unnorm_prob(self, x: Array) -> Array:
            return self.unnorm_prob_with_params(self.get_params(), x)

        def approx_pdf(self, rng: KeyArray, xs: Array, n_samples: int) -> Array:
            """Returns an approximated normalized density for each input point.

            This required computing the normalisation constant, which might be slow.
            """
            assert xs.ndim == 2 and xs.shape[1] == 1
            norm_constant = self.compute_norm_constant(rng, n_samples)
            unnorm_densities = vmap(self.unnorm_prob)(xs)
            return unnorm_densities / norm_constant

        def compute_norm_constant(self, rng: KeyArray, n_samples: int) -> Array:
            # Based on importance sampled approach described in
            # "Learning deep kernels for exponential family densities", Wenliang et al
            # Section 3.2
            samples = q0.sample(rng, n_samples)
            unnorm_densities = vmap(self.unnorm_prob)(samples)
            q0_densities = vmap(q0.prob)(samples)
            rs = unnorm_densities / q0_densities
            return rs.mean()

    return KernelExpFamily


@partial(jit, static_argnames=("p"))
def phi(x: Array, l: Array, p: int) -> Array:
    assert x.shape == () or x.shape == (1,), f"Shape was {x.shape}"
    # We use a finite set of basis functions to approximate functions in the
    # RKHS. The basis functions are taken from
    # "An explicit description of RKHSes of Gaussian RBK Kernels"; Steinwart 2006
    # Theorem 3, Equation 6.
    sigma = 1 / l
    js, sqrt_factorials = compute_js_and_sqrt_factorials(p)
    m1 = ((jnp.sqrt(2) * sigma * x) ** js) / sqrt_factorials
    m2 = jnp.exp(-((sigma * x) ** 2))
    return m1 * m2


def compute_js_and_sqrt_factorials(p: int) -> Tuple[Array, Array]:
    """Computes the square roots of 1...p factorial."""
    index_start, index_end = 1, p + 1
    js = jnp.arange(index_start, index_end)
    sqrt_factorials = [1.0]
    for j in range(index_start + 1, index_end):
        sqrt_factorials.append(sqrt_factorials[-1] * math.sqrt(j))
    return js, jnp.array(sqrt_factorials)
