from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import runners
import seaborn as sns
from jax import Array
from jax.random import KeyArray, PRNGKey
from pandas import DataFrame
from runners import ExperimentRunner

import figures
from composite_tests.bootstrapped_tests import (
    Bootstrap,
    parametric_bootstrap_test,
    wild_bootstrap_test,
)
from composite_tests.distributions import SampleableDist
from composite_tests.distributions.gaussian import Gaussian
from composite_tests.distributions.students_t import UnivariateT
from composite_tests.estimators import Estimator, TrueEstimator
from composite_tests.kernels import GaussianKernel
from composite_tests.ksd import KSDAnalyticEstimator, KSDStatistic

ParameterMode = Literal["est", "true"]
H = Literal["null", "alt"]

NULL_DIST = Gaussian(loc=0.4, scale=1.4)


class Result(NamedTuple):
    power: float
    theta_hats: Array


@dataclass
class Config:
    rng: KeyArray
    est_l: float
    test_l: float
    n: int
    parameter_mode: ParameterMode
    bootstrap: Bootstrap
    h: H
    alt_df: float = 2.0
    n_bootstrap_samples: int = 400
    n_repeats: int = 2000

    @property
    def name(self) -> str:
        return (
            f"{self.h}_"
            f"df{self.alt_df:.1f}_"
            f"el{self.est_l:.3f}_"
            f"tl{self.test_l:.3f}_"
            f"n{self.n}_"
            f"{self.bootstrap.value}_"
            f"bs{self.n_bootstrap_samples}_"
            f"p{self.parameter_mode}_"
            f"r{self.n_repeats}"
        )


def exp_bad_lengthscales_wild(runner: ExperimentRunner) -> None:
    _exp_bad_lengthscales(runner, bootstrap=Bootstrap.WILD)


def exp_bad_lengthscales_parametric(runner: ExperimentRunner) -> None:
    _exp_bad_lengthscales(runner, bootstrap=Bootstrap.PARAMETRIC)


def _exp_bad_lengthscales(runner: ExperimentRunner, bootstrap: Bootstrap) -> None:
    rng = PRNGKey(seed=80982344)
    n = 100
    parameter_modes: List[ParameterMode] = ["est", "true"]

    log_ls = [-2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    ls = [10**x for x in log_ls]
    for l in ls:
        for parameter_mode in parameter_modes:
            rng, rng_input = jax.random.split(rng)
            config = Config(rng_input, l, l, n, parameter_mode, bootstrap, h="null")
            name = f"bl_{config.name}"
            runner.queue(name, config)

    def plot(results: List[Tuple[Config, Result]]) -> None:
        figures.configure_matplotlib()
        plt.figure(figsize=(figures.THIRD_WIDTH * 2, figures.COMPACT_HEIGHT))
        power_ax = plt.gca()
        theta_hat_ax = power_ax.twinx()

        power_color = "C0"
        param_color = "C1"

        results = [(c, Result(*r)) for c, r, in results]
        df = DataFrame.from_records(
            [(c.est_l, c.parameter_mode, r.power, r.theta_hats) for c, r in results],
            columns=["l", "parameter", "power", "theta_hats"],
        )
        gb = df.groupby(["parameter"])
        for parameter_mode in gb.groups.keys():
            powers = gb.get_group(parameter_mode)[["l", "power"]].astype(float)
            if parameter_mode == "est":
                label = "composite test"
            else:
                label = "non-composite test"
            linestyle = "--" if parameter_mode == "est" else ":"
            powers.plot(
                x="l",
                y="power",
                label=label,
                ax=power_ax,
                color=power_color,
                linestyle=linestyle,
            )

            if parameter_mode == "est":
                theta_hats = gb.get_group(parameter_mode)[["l", "theta_hats"]]
                label = parameter_mode
                for l, ths in theta_hats.values:
                    stds = jnp.sqrt(ths[:, 1].reshape(-1))
                    x_vals = jnp.full(stds.shape, l)
                    theta_hat_ax.scatter(
                        x_vals, stds.reshape(-1), alpha=0.05, color=param_color, s=1
                    )

        power_ax.axhline(0.05, linestyle="-", color=power_color, linewidth=1)
        theta_hat_ax.axhline(
            NULL_DIST.scale, linestyle="-", color=param_color, linewidth=1
        )
        median_heuristic_l = 1.05
        theta_hat_ax.axvline(median_heuristic_l, color="black", zorder=-5)

        power_ax.set_zorder(2.1)
        theta_hat_ax.set_zorder(2.0)
        power_ax.patch.set_visible(False)

        figures.set_axis_color(power_ax, power_color)
        power_ax.spines["right"].set_visible(False)
        figures.set_axis_color(theta_hat_ax, param_color)

        power_ax.set_xscale("log")
        power_ax.set_xticks([0.01, 1.0, 100.0])
        power_ax.set_xlabel("lengthscale, $l$", **figures.squashed_label_params)
        power_ax.set_ylabel("type I error rate", **figures.squashed_label_params)
        power_ax.set_ylim(0.0, 0.16)
        power_ax.set_yticks([0.0, 0.15])
        theta_hat_ax.set_ylabel("estimated scale", **figures.squashed_label_params)
        power_ax.legend(**figures.squashed_legend_params)

        save_fig(runner)
        plt.close()

    runner.run(run_exp, plot)


def exp_grid(runner: ExperimentRunner) -> None:
    rng = PRNGKey(seed=80982345)
    n = 50

    ls = jnp.arange(1.0, 7.0, 0.5).tolist()
    hs: List[H] = ["null", "alt"]
    for est_l in ls:
        for test_l in ls:
            for h in hs:
                rng, rng_input = jax.random.split(rng)
                config = Config(
                    rng_input,
                    est_l,
                    test_l,
                    n,
                    parameter_mode="est",
                    alt_df=3.0,
                    bootstrap=Bootstrap.PARAMETRIC,
                    h=h,
                )
                name = f"bl_{config.name}"
                runner.queue(name, config)

    def plot(results: List[Tuple[Config, Result]]) -> None:
        # Height has to be smaller because the color bar takes up some width, so if
        # height is not the constraining factor it sticks out the top.
        plt.figure(figsize=(figures.FULL_WIDTH * 0.7, figures.FULL_WIDTH * 0.7 * 0.8))
        results = [(c, Result(*r)) for c, r, in results]
        df = DataFrame.from_records(
            [(c.est_l, c.test_l, c.h, r.power) for c, r in results],
            columns=["est l", "test l", "h", "power"],
        ).set_index(["est l", "test l"])
        alt_df = df[df["h"] == "alt"]
        # Remove entries where the type I error is too high.
        null_df = df[df["h"] == "null"]
        alt_df.loc[null_df["power"] > 0.07, "power"] = 0

        alt_df = alt_df.reset_index()
        alt_df["power"] = alt_df["power"] * 100
        alt_df = alt_df.round({"est l": 3, "test l": 3, "power": 0})

        alt_grid_df = alt_df.pivot(index="est l", columns="test l", values="power")
        ax = sns.heatmap(
            alt_grid_df, annot=True, fmt=".0f", square=True, cmap="crest_r"
        )
        ax.invert_yaxis()
        ax.set_xlabel("test l", **figures.squashed_label_params)
        ax.set_ylabel("estimator l", **figures.squashed_label_params)
        save_fig(runner)

    runner.run(run_exp, plot)


def run_exp(config: Config) -> Result:
    rng = config.rng
    null_family = Gaussian

    if config.h == "null":
        other_dist: SampleableDist = NULL_DIST
    elif config.h == "alt":
        other_dist = UnivariateT(df=config.alt_df)

    if config.parameter_mode == "est":
        estimator: Estimator = KSDAnalyticEstimator(
            GaussianKernel(config.est_l), null_family
        )
    elif config.parameter_mode == "true":
        assert isinstance(other_dist, Gaussian)
        estimator = TrueEstimator(jnp.array([other_dist.loc, other_dist.scale**2]))
    test_statistic = KSDStatistic(GaussianKernel(config.test_l), null_family)

    results = []
    theta_hats = []
    for repeat in range(config.n_repeats):
        rng, rng_input = jax.random.split(rng)
        ys = other_dist.sample(rng_input, config.n)
        rng, rng_input = jax.random.split(rng)
        if config.bootstrap == Bootstrap.PARAMETRIC:
            result = parametric_bootstrap_test(
                rng_input,
                ys,
                estimator,
                null_family,
                test_statistic,
                config.n_bootstrap_samples,
                save_null_distribution=False,
            )
        elif config.bootstrap == Bootstrap.WILD:
            result = wild_bootstrap_test(
                rng_input,
                ys,
                estimator,
                test_statistic,
                config.n_bootstrap_samples,
                save_null_distribution=False,
            )
        results.append(result.reject_null)
        theta_hats.append(result.theta_hat)
    power = sum(results) / len(results)
    return Result(power, jnp.stack(theta_hats))


def save_fig(runner: ExperimentRunner) -> None:
    figures.save_fig(f"bl_{runner.exp}")


if __name__ == "__main__":
    figures.configure_matplotlib()

    exp_funcs: Dict[str, Callable[[ExperimentRunner], None]] = {
        "bad_lengthscales_wild": exp_bad_lengthscales_wild,
        "bad_lengthscales_parametric": exp_bad_lengthscales_parametric,
        "grid": exp_grid,
    }
    runners.main(exp_funcs)
