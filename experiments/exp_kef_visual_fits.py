"""Experiment that compares the fit of KEF with different number of basis functions."""

from math import sqrt

import galaxy_dataset
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap
from jax.numpy import ndarray
from jax.random import PRNGKey
from tap import Tap

import figures
from composite_tests.bootstrapped_tests import (
    Bootstrap,
    parametric_bootstrap_test,
    wild_bootstrap_test,
)
from composite_tests.distributions.kef import kernel_exp_family
from composite_tests.kernels import IMQKernel, SumKernel
from composite_tests.ksd import KSDAnalyticEstimator, KSDStatistic
from composite_tests.rejection_sampler import LowAcceptanceError


def main(bootstrap: Bootstrap):
    rng = PRNGKey(seed=56789)

    figures.configure_matplotlib()
    plt.figure(figsize=(figures.HALF_WIDTH, figures.COMPACT_HEIGHT))

    kef_l = sqrt(2)
    kef_q0_std = 3.0
    ys, unnormalize = galaxy_dataset.load_galaxies()

    kernel = SumKernel([IMQKernel(l) for l in [0.6, 1.0, 1.2]])
    test_ps = [1, 2, 3, 4, 5, 25]

    xlims = (-7.5, 7.5)
    x_label = "galaxy velocity (km/s)"

    xs = jnp.linspace(xlims[0], xlims[1], 200).reshape(-1, 1)
    plt.hist(ys.reshape(-1), color="grey", alpha=0.5, bins=20, density=True)

    lines = []
    labels = []
    for p in test_ps:
        null = kernel_exp_family(p, kef_l, q0_std=kef_q0_std)
        test_stat = KSDStatistic(kernel, null)
        estimator = KSDAnalyticEstimator(kernel, null)

        try:
            rng, rng_input = jax.random.split(rng)
            if bootstrap == Bootstrap.PARAMETRIC:
                test_result = parametric_bootstrap_test(
                    rng_input,
                    ys,
                    estimator,
                    null,
                    test_stat,
                    n_bootstrap_samples=400,
                    save_null_distribution=False,
                )
            elif bootstrap == Bootstrap.WILD:
                test_result = wild_bootstrap_test(
                    rng_input,
                    ys,
                    estimator,
                    test_stat,
                    n_bootstrap_samples=500,
                    save_null_distribution=False,
                )

            est_null_dist = null(test_result.theta_hat)
            labels.append(f"$p={p}$")
            rng, rng_input = jax.random.split(rng)
            (line,) = plt.plot(
                xs.reshape(-1),
                est_null_dist.approx_pdf(rng_input, xs, n_samples=2000),
                label=labels[-1],
                linestyle="--" if test_result.reject_null else "-",
            )
            lines.append(line)

            print(
                f"p={p} "
                f"stat={test_result.test_statistic:.3f} "
                f"threshold={test_result.threshold:.3f} "
                f"reject_null={test_result.reject_null}"
            )

        except LowAcceptanceError as e:
            print(str(e))
            _plot_low_acceptance_error(xs, e)
            break

    half_i = len(test_ps) // 2
    l1 = plt.legend(
        lines[:half_i],
        labels[:half_i],
        loc="upper left",
        **figures.squashed_legend_params,
    )
    l2 = plt.legend(
        lines[half_i:],
        labels[half_i:],
        loc="upper right",
        **figures.squashed_legend_params,
    )
    plt.gca().add_artist(l1)
    plt.gca().add_artist(l2)

    plt.xlabel(x_label, **figures.squashed_label_params)
    x_ticks_normalized = [xlims[0] * 0.7, 0.0, xlims[1] * 0.7]
    x_ticks_unnormalized = [
        f"{unnormalize(jnp.array(x)):.0f}" for x in x_ticks_normalized
    ]
    plt.xticks(x_ticks_normalized, x_ticks_unnormalized)
    plt.yticks([])
    plt.xlim(*xlims)
    figures.save_fig(f"kvf_galaxies_{bootstrap.value}")
    plt.close()


def _plot_low_acceptance_error(xs: ndarray, e: LowAcceptanceError) -> None:
    proposal_pdf = lambda x: jnp.exp(e.proposal_log_pdf(x))
    target_pdf = lambda x: jnp.exp(e.target_log_pdf(x))
    plt.plot(xs.reshape(-1), vmap(proposal_pdf)(xs).reshape(-1), label="proposal pdf")
    plt.plot(xs.reshape(-1), vmap(target_pdf)(xs).reshape(-1), label="target pdf")


class Args(Tap):
    bootstrap: Bootstrap = Bootstrap.PARAMETRIC


if __name__ == "__main__":
    args = Args().parse_args()
    main(args.bootstrap)
