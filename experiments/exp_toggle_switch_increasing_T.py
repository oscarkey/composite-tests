from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import runners
from jax.random import KeyArray, PRNGKey
from pandas import DataFrame
from runners import ExperimentFunc, ExperimentRunner
from scipy.stats import gaussian_kde

import figures
from composite_tests.bootstrapped_tests import (
    Bootstrap,
    TestResult,
    parametric_bootstrap_test,
    wild_bootstrap_test,
)
from composite_tests.distributions.toggle_switch import (
    ToggleSwitchEstimator,
    small_toggle_switch,
)
from composite_tests.mmd import MMDStatistic

FIGURE_HEIGHT = 0.8 * figures.COMPACT_HEIGHT


@dataclass
class Config:
    rng: KeyArray
    null_T: int
    n: int
    seed: int
    bootstrap: Bootstrap = Bootstrap.PARAMETRIC

    @property
    def name(self) -> str:
        return f"T{self.null_T}_n{self.n}_s{self.seed}_{self.bootstrap.value}"


def exp_increasing_T(runner: ExperimentRunner) -> None:
    rng = PRNGKey(seed=1489048490)
    n = 400
    n_seeds = 10
    Ts = list(range(1, 71, 1))
    Ts = [10, 50, 100, 200]

    for T in Ts:
        for seed in range(n_seeds):
            rng, rng_input = jax.random.split(rng)
            config = Config(rng_input, T, n, seed)
            runner.queue(f"tsiT_{config.name}", config)

    def plot(results: List[Tuple[Config, TestResult]]) -> None:
        plt.figure(figsize=(figures.HALF_WIDTH, FIGURE_HEIGHT))
        ax = plt.gca()
        df = DataFrame.from_records(
            [(c.seed, c.null_T, r.reject_null) for c, r in results],
            columns=["seed", "T", "reject"],
        ).groupby(["T"])
        n_rejected = df.sum()["reject"]
        n_total = df.count()["reject"]
        power_df = n_rejected / n_total
        ts = power_df.reset_index()["T"].values

        cmap = plt.get_cmap("viridis_r", n_seeds)
        im = ax.imshow(power_df.values.reshape(1, -1), cmap=cmap, aspect="auto")
        tick_labels = [ts[0], 25, ts[-1]]
        tick_positions = [x - 0.5 for x in tick_labels]
        ax.set_xticks(tick_positions, labels=tick_labels)
        ax.set_xlim(tick_positions[0], tick_positions[-1])
        ax.set_yticks([])

        colorbar = plt.colorbar(im, ax=ax, aspect=2)
        colorbar.set_ticks([0, 1])
        ax.set_xlabel("T", **figures.squashed_label_params)
        figures.save_fig("tsiT_rejects")
        plt.close()

        # Plot one case where the null holds, and one where it doesn't.
        alt_c, alt_r = [(c, r) for c, r in results if c.null_T == 10][0]
        null_c, null_r = [(c, r) for c, r in results if c.null_T == 200][0]
        _plot_fit(null_c, null_r, alt_c, alt_r)

    runner.run(run_config, plot)


def _plot_fit(
    good_config: Config,
    good_result: TestResult,
    bad_config: Config,
    bad_result: TestResult,
) -> None:
    rng = good_config.rng
    rng, rng_input = jax.random.split(rng)
    ys = small_toggle_switch()().sample(rng_input, good_config.n)

    rng, rng_input = jax.random.split(rng)
    null_xs = small_toggle_switch(good_config.null_T).sample_with_params(
        rng_input, good_result.theta_hat, n=ys.shape[0]
    )

    rng, rng_input = jax.random.split(bad_config.rng)
    alt_xs = small_toggle_switch(bad_config.null_T).sample_with_params(
        rng_input, bad_result.theta_hat, n=ys.shape[0]
    )

    x_max = 1250.0
    x_vals = jnp.linspace(0.0, x_max, 120)
    bw_method = 0.1
    data_kde = gaussian_kde(ys.reshape(-1), bw_method=bw_method)
    null_kde = gaussian_kde(null_xs.reshape(-1), bw_method=bw_method)
    alt_kde = gaussian_kde(alt_xs.reshape(-1), bw_method=bw_method)

    plt.figure(figsize=(figures.HALF_WIDTH, FIGURE_HEIGHT))
    plt.fill_between(x_vals, data_kde(x_vals), label="true model", color="lightgrey")
    plt.plot(x_vals, alt_kde(x_vals), label=f"$T={bad_config.null_T}$", color="C1")
    plt.plot(x_vals, null_kde(x_vals), label=f"$T={good_config.null_T}$", color="C0")

    plt.xlim(0.0, x_max)
    plt.ylim(bottom=0.0)
    plt.xlabel("y", **figures.squashed_label_params)
    plt.yticks([])
    plt.legend(**figures.squashed_legend_params)

    figures.save_fig("tsiT_fits")
    plt.close()


def run_config(config: Config) -> TestResult:
    rng = config.rng

    null_family = small_toggle_switch(config.null_T)
    estimator = ToggleSwitchEstimator(config.null_T)
    statistic = MMDStatistic(estimator.kernel, null_family)

    rng, rng_input = jax.random.split(rng)
    ys = null_family().sample(rng_input, config.n)

    rng, rng_input = jax.random.split(rng)
    if config.bootstrap == Bootstrap.PARAMETRIC:
        return parametric_bootstrap_test(
            rng_input,
            ys,
            estimator,
            null_family,
            statistic,
            n_bootstrap_samples=200,
            save_null_distribution=True,
            parallel_samples=10,
        )
    elif config.bootstrap == Bootstrap.WILD:
        return wild_bootstrap_test(
            rng_input,
            ys,
            estimator,
            statistic,
            n_bootstrap_samples=400,
            save_null_distribution=True,
        )


if __name__ == "__main__":
    figures.configure_matplotlib()
    exp_funcs: Dict[str, ExperimentFunc] = {
        "increasing_T": exp_increasing_T,
    }
    runners.main(exp_funcs)
