from functools import wraps
from math import ceil
from typing import Callable, Iterable, ParamSpec, TypeVar, cast

import jax.numpy as jnp
from jax import Array, tree_util, vmap
from jax.random import KeyArray
from tqdm import tqdm

T = TypeVar("T")


def tree_concatenate(trees: Iterable[T]) -> T:
    """Concatenates the leaves of a list of pytrees to produce a single pytree."""
    leaves, treedefs = zip(*[tree_util.tree_flatten(tree) for tree in trees])
    grouped_leaves = zip(*leaves)
    result_leaves = [jnp.concatenate(l) for l in grouped_leaves]
    return cast(T, treedefs[0].unflatten(result_leaves))


def batch_vmap(
    f: Callable[[KeyArray], T], rngs: KeyArray, batch_size: int, progress: bool = False
) -> T:
    """Equivalent to vmap(f)(rngs), but vmaps only batch_size rngs at a time.

    This reduces memory usage.

    :param progress: If True, displays a progress bar.
    """
    n_batches = int(ceil(rngs.shape[0] / batch_size))

    batch_results: list[T] = []
    if progress:
        iterator = tqdm(range(n_batches))
    else:
        iterator = range(n_batches)
    for batch_i in iterator:
        batch_rngs = rngs[batch_i * batch_size : (batch_i + 1) * batch_size]
        batch_results.append(vmap(f)(batch_rngs))

    return tree_concatenate(batch_results)


P = ParamSpec("P")


def to_scalar(f: Callable[P, Array]) -> Callable[P, Array]:
    @wraps(f)
    def f2(*args: P.args, **kwargs: P.kwargs) -> Array:
        return f(*args, **kwargs).reshape(())

    return f2
