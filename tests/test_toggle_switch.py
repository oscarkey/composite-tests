import jax
from jax.random import PRNGKey

from composite_tests.distributions.toggle_switch import small_toggle_switch


def test__small_toggle_switch__sample__small_model__does_not_crash():
    rng = PRNGKey(seed=8028340324)
    dist = small_toggle_switch(T=30)()
    rng, rng_input = jax.random.split(rng)
    ys = dist.sample(rng_input, n=300)
    assert ys.shape == (300, 1)
