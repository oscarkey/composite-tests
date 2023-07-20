from functools import partial
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.interpreters.batching import BatchTracer
from jax.random import KeyArray

from composite_tests.distributions import SampleableAndNormalizedDist
from composite_tests.extra_types import Scalar


def sample(
    rng: KeyArray,
    proposal: SampleableAndNormalizedDist,
    target_log_pdf: Callable[[Array], Array],
    n: int,
) -> tuple[Array, Scalar]:
    """Samples from a target distribution using rejection sampling.

    :param proposal_sample: (rng, n) -> samples [nxd], returns n samples from proposal
    :param proposal_log_pdf: sample [d] -> log density [1], can be unnormalized
    :param target_log_pdf: sample [d] -> log density [1], can be unnormalized
    """
    # In this method we remember that we might be operating in batched mode, i.e we
    # are inside a vmap call and several sets of samples in parallel.
    # Thus we use a special _get_min_across_batch() method, and have to avoid
    # boolean indexing below.

    n_tried = jnp.array(0)
    n_accepted = jnp.array(0)

    all_samples: List[Array] = []
    accept_masks: List[Array] = []

    log_c_est = jnp.log(jnp.array(1.0001))

    while _get_min_across_batch(n_accepted) < n:
        rng, rng_input = jax.random.split(rng)
        samples = proposal.sample(rng_input, n)

        proposal_log_ds, target_log_ds = _compute_log_densities(
            proposal.log_prob, target_log_pdf, samples
        )

        rng, rng_input = jax.random.split(rng)
        accept_mask, n_accepted_this_iter, log_c_est = _compute_accept_mask(
            rng_input, target_log_ds, proposal_log_ds, log_c_est
        )

        all_samples.append(samples)
        accept_masks.append(accept_mask)

        n_tried += n
        n_accepted += n_accepted_this_iter

        running_acceptance_rate = _get_min_across_batch(n_accepted) / n_tried
        if running_acceptance_rate < 0.01 and n_tried >= n * 3:
            raise LowAcceptanceError(
                f"Acceptance rate is low, giving up after {n_tried} tries. "
                f"Min rate = {100 * running_acceptance_rate:.2f}%",
                target_log_pdf,
                proposal.log_prob,
                jnp.exp(log_c_est),
            )

    joined_samples = jnp.concatenate(all_samples)
    joined_masks = jnp.concatenate(accept_masks)
    # To support batching JAX needs to know the size of each array in advance.
    # Thus we can't simply do a boolean select of the samples by the mask.
    # Instead we make a list of indices. For indices we want, we put the actual
    # index. For indices we don't want we put a large number (shape[0]+10). We then
    # sort this list of indices, cut it to the first n, and select the samples at
    # these indices (which is okay because JAX knows there are exactly n). This
    # selects n acceptable samples for each element of the batch because all the
    # unacceptable samples have "large indices" and so end up at the end of the
    # list, and we know that each element of the batch has at least n acceptable
    # samples.
    indices_to_accept = jnp.where(
        joined_masks,
        jnp.arange(joined_masks.shape[0]),
        jnp.full(joined_masks.shape[0], joined_masks.shape[0] + 10),
    )
    n_indices_to_accept = jnp.sort(indices_to_accept)[:n]
    accepted_samples = joined_samples[n_indices_to_accept]

    acceptance_rate = _get_min_across_batch(n_accepted) / n_tried

    return accepted_samples, acceptance_rate


@partial(jit, static_argnames=("proposal_log_pdf", "target_log_pdf"))
def _compute_log_densities(
    proposal_log_pdf: Callable[[Array], Array],
    target_log_pdf: Callable[[Array], Array],
    samples: Array,
) -> Tuple[Array, Array]:
    proposal_log_pds = vmap(proposal_log_pdf, in_axes=0)(samples)
    target_log_pds = vmap(target_log_pdf, in_axes=0)(samples)
    return proposal_log_pds, target_log_pds


@jit
def _compute_accept_mask(
    rng_input: Array,
    target_log_densities: Array,
    proposal_log_densities: Array,
    log_c_est: Array,
) -> Tuple[Array, Array, Array]:
    proposal_log_densities = proposal_log_densities.reshape(-1)
    target_log_densities = target_log_densities.reshape(-1)

    log_ratio = target_log_densities - proposal_log_densities
    log_c_est = jnp.maximum(log_c_est, log_ratio.max())

    reject_p = jnp.exp(log_ratio - log_c_est)
    accept_mask = jax.random.uniform(rng_input, shape=log_ratio.shape) < reject_p
    n_accepted = accept_mask.sum()

    return accept_mask, n_accepted, log_c_est


def _get_min_across_batch(x: Array) -> Array:
    if isinstance(x, BatchTracer):
        # If the array is actually a batch of arrays, retrieve the batch as a single
        # array and take the minimum.
        return x.val.min()
    else:
        return x


class LowAcceptanceError(Exception):
    def __init__(
        self,
        message: str,
        target_log_pdf: Callable[[Array], Array],
        proposal_log_pdf: Callable[[Array], Array],
        c: Array,
    ) -> None:
        super().__init__(message)
        self.target_log_pdf = target_log_pdf
        self.proposal_log_pdf = proposal_log_pdf
        self.c = c
