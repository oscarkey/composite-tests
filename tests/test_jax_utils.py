import chex
import jax.numpy as jnp
from jax import Array, vmap
from jax.random import KeyArray, PRNGKey, split, uniform

from composite_tests.jax_utils import batch_vmap, tree_concatenate


@chex.dataclass
class ExamplePytree:
    a: Array
    b: Array


def test__tree_concatenate__correctly_concatenates_leaves():
    one = ExamplePytree(a=jnp.array([1]), b=jnp.array([2]))
    two = ExamplePytree(a=jnp.array([10]), b=jnp.array([20]))

    cat = tree_concatenate([one, two])

    assert jnp.allclose(cat.a, jnp.array([1, 10]))
    assert jnp.allclose(cat.b, jnp.array([2, 20]))


def test__batch_vmap__result_same_as_single_vmap():
    rngs = split(PRNGKey(seed=1231245), num=100)

    def f(rng: KeyArray) -> tuple[Array, Array]:
        rng1, rng2 = split(rng)
        return uniform(rng1), uniform(rng2)

    expected = vmap(f)(rngs)
    batched = batch_vmap(f, rngs, batch_size=3)
    batched_with_progress = batch_vmap(f, rngs, batch_size=3, progress=True)

    assert jnp.allclose(batched[0], expected[0])
    assert jnp.allclose(batched[1], expected[1])
    assert jnp.allclose(batched_with_progress[0], expected[0])
    assert jnp.allclose(batched_with_progress[1], expected[1])
