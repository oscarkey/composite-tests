from typing import Any, Callable, Optional, TypeVar

import jax
import jax.numpy as jnp
import optax
from jax import Array, lax, vmap
from jax.random import KeyArray
from jax.tree_util import tree_map
from jaxopt import OptaxSolver
from jaxopt._src.base import OptStep
from jaxopt._src.optax_wrapper import OptaxState
from optax import GradientTransformation

P = TypeVar("P")
LossFunction = Callable[[KeyArray, P], Array]


def random_restart_optimizer(
    rng: KeyArray,
    optimizer: GradientTransformation,
    loss: LossFunction,
    sample_theta_init: Callable[[KeyArray], P],
    iterations: int,
    n_initial_locations: int,
    n_optimized_locations: int,
    params_to_fix: Optional[Any] = None,
) -> P:
    rng, *rng_inputs = jax.random.split(rng, num=n_initial_locations + 1)
    init_thetas = vmap(sample_theta_init)(jnp.array(rng_inputs))

    rng, *rng_inputs = jax.random.split(rng, num=n_initial_locations + 1)
    init_losses = vmap(loss, in_axes=(0, 0))(jnp.array(rng_inputs), init_thetas)
    best_init_indices = init_losses.argsort()[:n_optimized_locations]

    best_init_params = _index_params(init_thetas, best_init_indices)
    rng, *rng_inputs = jax.random.split(rng, num=n_optimized_locations + 1)
    opt_params = vmap(run_optimizer, in_axes=(0, None, None, 0, None, None))(
        jnp.array(rng_inputs),
        optimizer,
        loss,
        best_init_params,
        iterations,
        params_to_fix,
    )

    rng, *rng_inputs = jax.random.split(rng, num=n_optimized_locations + 1)
    final_losses = vmap(loss)(jnp.array(rng_inputs), opt_params)
    return _index_params(opt_params, final_losses.argsort()[0])


def _index_params(params_pytree: P, indices: Array) -> P:
    return tree_map(lambda p: p[indices], params_pytree)


def run_optimizer(
    rng: KeyArray,
    optimizer: GradientTransformation,
    loss: LossFunction,
    theta_init: P,
    iterations: int,
    params_to_fix: Optional[Any] = None,
) -> P:
    def f(params: P, data: KeyArray) -> Array:
        local_rng = data
        return loss(local_rng, params)

    if params_to_fix:
        grad_transform = optax.multi_transform(
            {False: optimizer, True: optax.set_to_zero()},
            params_to_fix,
        )
    else:
        grad_transform = optimizer
    solver = OptaxSolver(f, grad_transform, maxiter=iterations, jit=True)
    rng, rng_input = jax.random.split(rng)
    return _run_solver(rng_input, solver, theta_init, iterations)


def _run_solver(
    rng: KeyArray, solver: OptaxSolver, theta_init: P, iterations: int
) -> P:
    init_state = solver.init_state(theta_init)

    def update(step: OptStep, local_rng: KeyArray) -> tuple[P, OptaxState]:
        new_step = solver.update(step.params, step.state, data=local_rng)
        return new_step, None

    init_step = OptStep(theta_init, init_state)
    rng_inputs = jax.random.split(rng, num=iterations)
    (final_params, _), _ = lax.scan(update, init_step, rng_inputs)
    return final_params
