from __future__ import annotations

from abc import ABC
from dataclasses import asdict
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import optax
from chex import dataclass
from jax import Array, jit, vmap
from jax.lax import scan, stop_gradient
from jax.random import KeyArray, truncated_normal

import composite_tests.mmd as mmd
from composite_tests.distributions import SampleableDist
from composite_tests.estimators import Estimator
from composite_tests.extra_types import Scalar
from composite_tests.kernels import GaussianKernel, Kernel, SumKernel
from composite_tests.optimizers import random_restart_optimizer

a = lambda x: jnp.array(x)


@dataclass(frozen=True, eq=True)
class TSParams(ABC):
    alpha_1: Scalar
    alpha_2: Scalar
    beta_1: Scalar
    beta_2: Scalar
    mu: Scalar
    sigma: Scalar
    gamma: Scalar
    kappa_1: Scalar
    kappa_2: Scalar
    delta_1: Scalar
    delta_2: Scalar


@dataclass(frozen=True, eq=True)
class RawParams(TSParams):
    @staticmethod
    def from_array(params: Array) -> RawParams:
        return RawParams(
            alpha_1=params[0],
            alpha_2=params[1],
            beta_1=params[2],
            beta_2=params[3],
            mu=params[4],
            sigma=params[5],
            gamma=params[6],
            kappa_1=params[7],
            kappa_2=params[8],
            delta_1=params[9],
            delta_2=params[10],
        )

    def transform(self) -> TransformedParams:
        return TransformedParams(
            alpha_1=jnp.log(self.alpha_1),
            alpha_2=jnp.log(self.alpha_2),
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            mu=jnp.log(self.mu),
            sigma=jnp.log(self.sigma),
            gamma=self.gamma,
            kappa_1=self.kappa_1,
            kappa_2=self.kappa_2,
            delta_1=self.delta_1,
            delta_2=self.delta_2,
        )

    def perturb(self, **kwargs: Dict[str, Scalar]) -> RawParams:
        current = asdict(self)
        for param, update in kwargs.items():
            current[param] += update
        return RawParams(**current)


@dataclass(frozen=True, eq=True)
class TransformedParams(TSParams):
    def untransform(self) -> RawParams:
        return RawParams(
            alpha_1=jnp.exp(self.alpha_1),
            alpha_2=jnp.exp(self.alpha_2),
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            mu=jnp.exp(self.mu),
            sigma=jnp.exp(self.sigma),
            gamma=self.gamma,
            kappa_1=self.kappa_1,
            kappa_2=self.kappa_2,
            delta_1=self.delta_1,
            delta_2=self.delta_2,
        )


small_model_default_params = RawParams(
    alpha_1=a(22.0),
    alpha_2=a(12.0),
    beta_1=a(4.0),
    beta_2=a(4.5),
    mu=a(325.0),
    sigma=a(0.25),
    gamma=a(0.15),
    kappa_1=a(1.0),
    kappa_2=a(1.0),
    delta_1=a(0.03),
    delta_2=a(0.03),
).transform()

# We don't optimize the kappas or deltas for the small model, so just sample them at
# their true values using a range size of zero.
_small_model_param_range = (
    jnp.array([0.01, 0.01, 0.01, 0.01, 250.0, 0.01, 0.01, 1.0, 1.0, 0.03, 0.03]),
    jnp.array([50.0, 50.0, 5.0, 5.0, 450.0, 0.5, 0.4, 1.0, 1.0, 0.03, 0.03]),
)


def sample_initial_params(rng: KeyArray) -> TransformedParams:
    lower, upper = _small_model_param_range
    unparam_theta = jax.random.uniform(
        rng, minval=lower, maxval=upper, shape=lower.shape
    )
    return RawParams.from_array(unparam_theta).transform()


def small_toggle_switch(T: int = 300):
    @dataclass
    class SmallToggleSwitch(SampleableDist[TransformedParams, Array]):
        params: TransformedParams = small_model_default_params

        @staticmethod
        def sample_with_params(
            rng: KeyArray, params: TransformedParams, n: int
        ) -> Array:
            rng_inputs = jax.random.split(rng, n)
            samples = vmap(_small_model, in_axes=(0, None, None))(rng_inputs, params, T)
            return samples.reshape((n, 1))

        def get_params(self) -> TransformedParams:
            return self.params

        @staticmethod
        def sample_initial_params(rng: KeyArray) -> TransformedParams:
            lower, upper = _small_model_param_range
            unparam_theta = jax.random.uniform(
                rng, minval=lower, maxval=upper, shape=lower.shape
            )
            return RawParams.from_array(unparam_theta).transform()

    return SmallToggleSwitch


State = Array


@partial(jit, static_argnames=("T"))
def _small_model(rng: KeyArray, params: TransformedParams, T: int) -> Array:
    p = params.untransform()

    def step_function(state: State, rng: KeyArray) -> tuple[State, None]:
        u_t, v_t = state[0], state[1]
        rng, rng_input = jax.random.split(rng)
        u_mean = u_t + p.alpha_1 / (1 + v_t**p.beta_1) - (p.kappa_1 + p.delta_1 * u_t)
        u_next = _truncated_normal(rng_input, loc=u_mean, std=jnp.array(0.5))
        rng, rng_input = jax.random.split(rng)
        v_mean = v_t + p.alpha_2 / (1 + u_t**p.beta_2) - (p.kappa_2 + p.delta_2 * v_t)
        v_next = _truncated_normal(rng_input, loc=v_mean, std=jnp.array(0.5))

        return jnp.array([u_next, v_next]), None

    initial_state = jnp.array([10.0, 10.0])
    rng, *rng_inputs = jax.random.split(rng, num=1 + T)
    final_state, _ = scan(step_function, initial_state, jnp.array(rng_inputs))
    u_T, v_T = final_state[0], final_state[1]

    rng, rng_input = jax.random.split(rng)
    return _truncated_normal(
        rng_input,
        loc=p.mu + u_T,
        std=p.mu * p.sigma / (u_T**p.gamma),
    )


def _truncated_normal(rng: KeyArray, loc: Array, std: Array) -> Array:
    """Samples from a normal distribution, truncated below zero."""
    lower = stop_gradient((0.0 - loc) / std)
    upper = stop_gradient((1200.0 - loc) / std)
    return loc + std * truncated_normal(rng, lower, upper)


class ToggleSwitchEstimator(Estimator):
    def __init__(self, T: int) -> None:
        self.T = T
        self.kernel = SumKernel(
            [
                GaussianKernel(20.0),
                GaussianKernel(40.0),
                GaussianKernel(80.0),
                GaussianKernel(100.0),
                GaussianKernel(130.0),
                GaussianKernel(200.0),
                GaussianKernel(400.0),
                GaussianKernel(800.0),
                GaussianKernel(1000.0),
            ]
        )

    def __call__(self, rng: KeyArray, ys: Array) -> TransformedParams:
        return _estimate_toggle_switch(rng, self.kernel, self.T, ys)

    @property
    def name(self) -> str:
        return "mmd"


@partial(jit, static_argnames=("kernel", "T"))
def _estimate_toggle_switch(
    rng: KeyArray,
    kernel: Kernel,
    T: int,
    ys: Array,
) -> TransformedParams:
    params_to_fix = TransformedParams(
        alpha_1=a(False),
        alpha_2=a(False),
        beta_1=a(False),
        beta_2=a(False),
        mu=a(False),
        sigma=a(False),
        gamma=a(False),
        kappa_1=a(True),
        kappa_2=a(True),
        delta_1=a(True),
        delta_2=a(True),
    )

    def loss(rng2: KeyArray, ys: Array, p: TransformedParams) -> Array:
        xs = small_toggle_switch(T).sample_with_params(rng2, p, n=ys.shape[0])
        return mmd.v_stat(xs, ys, kernel)

    rng, rng_input = jax.random.split(rng)
    return random_restart_optimizer(
        rng_input,
        optax.adam(learning_rate=0.04),
        loss=lambda r, p: loss(r, ys, p),
        sample_theta_init=sample_initial_params,
        iterations=300,
        n_initial_locations=500,
        n_optimized_locations=15,
        params_to_fix=params_to_fix,
    )
