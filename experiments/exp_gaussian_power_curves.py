from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import runners
from jax import Array
from jax.numpy import ndarray
from jax.random import KeyArray, PRNGKey
from labellines import labelLine
from pandas import DataFrame
from runners import ExperimentRunner
from scipy.stats import anderson, normaltest, shapiro
from statsmodels.stats.diagnostic import lilliefors

import composite_tests.kernels as kernels
import figures
from composite_tests.bootstrapped_tests import (
    Bootstrap,
    TestStatistic,
    WildTestStatistic,
    parametric_bootstrap_test,
    wild_bootstrap_test,
)
from composite_tests.distributions import SampleableDist
from composite_tests.distributions.gaussian import (
    Gaussian,
    GaussianFixedScaleMMDEstimator,
    GaussianMMDEstimator,
    gaussian_fixed_scale,
    mvn_fixed_cov,
)
from composite_tests.distributions.students_t import MultivariateT, UnivariateT
from composite_tests.estimators import Estimator
from composite_tests.kernels import GaussianKernel, Kernel
from composite_tests.ksd import KSDAnalyticEstimator, KSDStatistic
from composite_tests.mmd import MMDStatistic

N_REPEATS = 2000
null_covar = lambda d: jnp.eye(d)


@dataclass
class Test(ABC):
    props: Dict

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(
        self, rng: KeyArray, ys: ndarray, use_cached_median_heuristic=False
    ) -> bool:
        pass


@dataclass
class Ours(Test):
    null_family: type[SampleableDist]
    lengthscale: Union[float, Literal["m"]]
    estimator: Callable[[float], Estimator]
    test_statistic: Callable[[float], TestStatistic]
    d: int
    bootstrap: Bootstrap
    m: Union[int, Literal["n"]] = "n"
    parametric_bootstrap_samples: int = 300
    wild_bootstrap_samples: int = 500
    median_heuristic_l: Optional[float] = None

    def __post_init__(self) -> None:
        if self.m != "n":
            raise NotImplementedError

        self._estimator: Optional[Estimator] = None
        self._test_statistic: Optional[TestStatistic] = None

    @property
    def name(self) -> str:
        l_str = "m" if self.lengthscale == "m" else f"{self.lengthscale:.2f}"

        # We construct an estimator and test statistic temporily to get the name :(
        estimator_name = self.estimator(1.0).name
        test_statistic_name = self.test_statistic(1.0).name

        name = (
            f"{estimator_name}_"
            f"{test_statistic_name}_"
            f"{self.bootstrap.value}_"
            f"kl{l_str}_"
            f"m{self.m}_"
            f"pbs{self.parametric_bootstrap_samples}_"
            f"wbs{self.wild_bootstrap_samples}"
        )

        return name

    def get_estimator(self, ys: ndarray, use_cache: bool) -> Estimator:
        if self._estimator is None or not use_cache:
            if self.lengthscale == "m":
                l = kernels.median_heuristic(ys).item()
                self.median_heuristic_l = l
            else:
                l = self.lengthscale

            self._estimator = self.estimator(l)

        return self._estimator

    def get_test_statistic(self, ys: ndarray, use_cache: bool) -> TestStatistic:
        if self._test_statistic is None or not use_cache:
            if self.lengthscale == "m":
                l = kernels.median_heuristic(ys).item()
                self.median_heuristic_l = l
            else:
                l = self.lengthscale

            self._test_statistic = self.test_statistic(l)

        return self._test_statistic

    def run(
        self, rng: KeyArray, ys: ndarray, use_cached_median_heuristic=False
    ) -> bool:
        test_statistic = self.get_test_statistic(ys, use_cached_median_heuristic)
        if self.bootstrap == Bootstrap.WILD:
            assert isinstance(test_statistic, WildTestStatistic)
            result = wild_bootstrap_test(
                rng,
                ys,
                self.get_estimator(ys, use_cached_median_heuristic),
                test_statistic,
                n_bootstrap_samples=self.wild_bootstrap_samples,
                save_null_distribution=False,
            )
        elif self.bootstrap == Bootstrap.PARAMETRIC:
            result = parametric_bootstrap_test(
                rng,
                ys,
                self.get_estimator(ys, use_cached_median_heuristic),
                self.null_family,
                test_statistic,
                self.parametric_bootstrap_samples,
                save_null_distribution=False,
            )
        return result.reject_null


class ShapiroWilks(Test):
    @property
    def name(self) -> str:
        return "sw"

    def run(
        self, rng: KeyArray, ys: ndarray, use_cached_median_heuristic=False
    ) -> bool:
        _, p = shapiro(ys)
        return p <= 0.05


class Lilliefors(Test):
    @property
    def name(self) -> str:
        return "lilliefors"

    def run(
        self, rng: KeyArray, ys: ndarray, use_cached_median_heuristic=False
    ) -> bool:
        assert ys.shape[1] == 1
        _, p = lilliefors(ys.reshape(-1))
        return p <= 0.05


class Anderson(Test):
    @property
    def name(self) -> str:
        return "anderson"

    def run(
        self, rng: KeyArray, ys: ndarray, use_cached_median_heuristic=False
    ) -> bool:
        assert ys.shape[1] == 1
        statistic, thresholds, sig_levels = anderson(ys.reshape(-1), dist="norm")
        threshold = [t for t, l in zip(thresholds, sig_levels) if l == 5.0]
        assert len(threshold) == 1
        return statistic > threshold[0]


class DAgostino(Test):
    @property
    def name(self) -> str:
        return "dagostino"

    def run(
        self, rng: KeyArray, ys: ndarray, use_cached_median_heuristic=False
    ) -> bool:
        assert ys.shape[1] == 1
        _, p = normaltest(ys.reshape(-1))
        return p <= 0.05


@dataclass
class Config:
    rng: KeyArray
    other_distribution: SampleableDist[Any, Array]
    x_val: float
    n: int
    d: int
    test: Test
    seed_i: int = 0

    @property
    def name(self) -> str:
        return f"s{self.seed_i}_r{N_REPEATS}_n{self.n}_d{self.d}_{self.test.name}"


@dataclass
class Result:
    power: float
    median_heuristic_l: Optional[float]


def _k(l: float) -> Kernel:
    return GaussianKernel(l)


def exp_increasing_n_null_ksd(runner: ExperimentRunner) -> None:
    return _exp_increasing_n(runner, scenario="null", disc="ksd")


def exp_increasing_n_alt_ksd(runner: ExperimentRunner) -> None:
    return _exp_increasing_n(runner, scenario="alt", disc="ksd")


def exp_increasing_n_null_mmd(runner: ExperimentRunner) -> None:
    return _exp_increasing_n(runner, scenario="null", disc="mmd")


def _exp_increasing_n(
    runner: ExperimentRunner,
    scenario: Literal["null", "alt"],
    disc: Literal["mmd", "ksd"],
) -> None:
    rng = PRNGKey(seed=890834)
    n_seeds = 4
    d = 1
    obs_mean = 1.3
    obs_var = null_covar(d).item()
    alt_df = 2.0

    null_family = Gaussian
    if disc == "ksd":
        estimator: Callable[[float], Estimator] = lambda l: KSDAnalyticEstimator(
            _k(l), null_family
        )
        test_statistic: Callable[[float], WildTestStatistic] = lambda l: KSDStatistic(
            _k(l), null_family
        )
    elif disc == "mmd":
        estimator = lambda l: GaussianMMDEstimator(_k(l))
        test_statistic = lambda l: MMDStatistic(_k(l), null_family)

    ns = [10, 50, 100, 200, 300, 400, 500, 600, 1000]

    tests = [
        Ours(
            lengthscale="m",
            estimator=estimator,
            test_statistic=test_statistic,
            d=d,
            null_family=null_family,
            bootstrap=Bootstrap.PARAMETRIC,
            props={"label": "parametric", "color": figures.PARAMETRIC_COLOR},
        ),
        Ours(
            lengthscale="m",
            estimator=estimator,
            test_statistic=test_statistic,
            d=d,
            null_family=null_family,
            bootstrap=Bootstrap.WILD,
            props={"label": "wild", "color": figures.WILD_COLOR},
        ),
    ]

    for n in ns:
        for test in tests:
            rng, *rng_inputs = jax.random.split(rng, num=n_seeds + 1)
            for seed_i, rng_input in enumerate(rng_inputs):
                if scenario == "null":
                    other_dist: SampleableDist = Gaussian(
                        loc=obs_mean, scale=jnp.sqrt(obs_var)
                    )
                    other_name = f"v{obs_var:.2f}"
                elif scenario == "alt":
                    other_dist = UnivariateT(alt_df)
                    other_name = f"df{alt_df:.2f}"
                config = Config(rng_input, other_dist, n, n, d, test, seed_i=seed_i)
                name = f"gpc_inc_n_{config.name}_{other_name}"
                runner.queue(name, config)

    def plot(results: List[Tuple[Config, Result]]) -> None:
        plt.figure(figsize=(figures.HALF_WIDTH, figures.COMPACT_HEIGHT))
        max_power = 0.0
        for test in tests:
            test_rs = [(c, r) for c, r in results if c.test is test]
            df = DataFrame.from_records(
                [(c.x_val, c.seed_i, r.power) for c, r in test_rs],
                columns=["n", "seed", "power"],
            )
            df = df.groupby("n")
            means = df.mean()["power"]
            max_power = max(means.values.max(), max_power)
            if n_seeds == 1:
                plt.plot(means.keys(), means.values, **test.props)
            else:
                sems = df.sem()["power"]
                plt.errorbar(means.keys(), means.values, yerr=sems.values, **test.props)

        plt.xlabel("number of observations, n", **figures.squashed_label_params)
        plt.xlim(left=min(ns))
        plt.xticks([min(ns), 250, 500, 750, 1000])
        plt.yticks([0.0, round(max_power, 2)])
        plt.axhline(0.05, linestyle="--", color="black")
        plt.legend(ncol=2, **figures.squashed_legend_params)

        if scenario == "null":
            plt.ylabel("type I error rate", **figures.squashed_label_params)
            plt.ylim(bottom=0.0, top=max_power + 0.01)
        else:
            plt.ylabel("power", **figures.squashed_label_params)
            plt.ylim(bottom=0.0, top=1.0)

        save_fig(runner)
        plt.close()

    runner.run(run_exp, plot)


def exp_mmd_vs_ksd(runner: ExperimentRunner) -> None:
    rng = PRNGKey(seed=890834)
    d = 1
    n_seeds = 4
    ns = [10, 200]
    null_scale = jnp.array(1.0)
    null_family = gaussian_fixed_scale(null_scale)
    variances = jnp.arange(0.2, 3.0, 0.2)

    tests = [
        Ours(
            lengthscale="m",
            estimator=lambda l: GaussianFixedScaleMMDEstimator(_k(l), null_scale),
            test_statistic=lambda l: MMDStatistic(_k(l), null_family),
            d=d,
            null_family=null_family,
            bootstrap=Bootstrap.PARAMETRIC,
            props={"label": "MMD", "color": figures.MMD_COLOR},
        ),
        Ours(
            lengthscale="m",
            estimator=lambda l: KSDAnalyticEstimator(_k(l), null_family),
            test_statistic=lambda l: KSDStatistic(_k(l), null_family),
            d=d,
            null_family=null_family,
            bootstrap=Bootstrap.PARAMETRIC,
            props={"label": "KSD", "color": figures.KSD_COLOR},
        ),
    ]

    for n in ns:
        for variance in variances:
            for test in tests:
                rng, *rng_inputs = jax.random.split(rng, num=n_seeds + 1)
                for seed_i, rng_input in enumerate(rng_inputs):
                    other_dist = Gaussian(loc=jnp.array(0.4), scale=jnp.sqrt(variance))
                    config = Config(
                        rng_input,
                        other_dist,
                        variance.item(),
                        n,
                        d,
                        test,
                        seed_i=seed_i,
                    )
                    name = f"gpc_vs_{config.name}_v{variance:.2f}"
                    runner.queue(name, config)

    def plot(results: List[Tuple[Config, Result]]) -> None:
        plt.figure(figsize=(figures.HALF_WIDTH, figures.COMPACT_HEIGHT))

        for test in tests:
            test_rs = [(c, r) for c, r in results if c.test is test]
            df = DataFrame.from_records(
                [(c.x_val, c.n, c.seed_i, r.power) for c, r in test_rs],
                columns=["var", "n", "seed", "power"],
            )
            for n in ns:
                df_for_n = df[df["n"] == n]
                df_for_n = df_for_n.groupby("var")
                means = df_for_n.mean()["power"]
                sems = df_for_n.sem()["power"]

                if n == ns[0]:
                    plt.errorbar(
                        means.keys(),
                        means.values,
                        yerr=sems.values,
                        linestyle="-",
                        color=test.props["color"],
                        label=test.props["label"],
                    )
                else:
                    plt.errorbar(
                        means.keys(),
                        means.values,
                        yerr=sems.values,
                        linestyle="--",
                        color=test.props["color"],
                    )

        plt.xlabel("variance, $\\sigma^2$", **figures.squashed_label_params)
        plt.ylabel("power", **figures.squashed_label_params)
        plt.xlim(variances[0], variances[-1])
        plt.ylim(0, 1.05)
        plt.yticks([0, 1])
        plt.axhline(0.05, linestyle="--", color="black")
        plt.legend(loc=(0.7, 0.5), **figures.squashed_legend_params)

        save_fig(runner)
        plt.close()

    runner.run(run_exp, plot)


def exp_increasing_d_parametric(runner: ExperimentRunner) -> None:
    return _exp_increasing_d(runner, bootstrap=Bootstrap.PARAMETRIC)


def exp_increasing_d_wild(runner: ExperimentRunner) -> None:
    return _exp_increasing_d(runner, bootstrap=Bootstrap.WILD)


def _exp_increasing_d(runner: ExperimentRunner, bootstrap: Bootstrap) -> None:
    rng = PRNGKey(seed=901908094)
    n = 50
    ds = [2] + list(range(5, 51, 5))
    dfs = [3.0, 5.0]

    for d in ds:
        mean = jnp.array([0.2 for _ in range(d)])
        null_cov = jnp.eye(d)
        null_dist = mvn_fixed_cov(null_cov)

        test = Ours(
            lengthscale="m",
            estimator=lambda l, dist=null_dist: KSDAnalyticEstimator(_k(l), dist),
            test_statistic=lambda l, dist=null_dist: KSDStatistic(_k(l), dist),
            d=d,
            null_family=null_dist,
            bootstrap=bootstrap,
            props={},
        )

        rng, rng_input = jax.random.split(rng)
        null_config = Config(
            rng_input,
            other_distribution=null_dist(mean),
            x_val=d,
            n=n,
            d=d,
            test=test,
        )
        runner.queue(f"gpc_inc_d_{null_config.name}_null", null_config)

        for df in dfs:
            rng, rng_input = jax.random.split(rng)
            alt_config = Config(
                rng_input,
                other_distribution=MultivariateT(df, mean, null_cov),
                x_val=d,
                n=n,
                d=d,
                test=test,
            )
            runner.queue(f"gpc_inc_d_{alt_config.name}_altdf{df:.1f}", alt_config)

    def plot(results: List[Tuple[Config, Result]]) -> None:
        plt.figure(figsize=(figures.HALF_WIDTH, figures.COMPACT_HEIGHT))

        data_frame = DataFrame.from_records(
            [
                (
                    c.d,
                    r.power,
                    cast(MultivariateT, c.other_distribution).params.df
                    if isinstance(c.other_distribution, MultivariateT)
                    else "null",
                )
                for c, r in results
            ],
            columns=["d", "power", "df"],
        )
        have_labelled_alt = False
        for df, rows in data_frame.groupby("df", sort=False):
            if df == "null":
                label = "$H^C_0$"
                color = figures.KSD_COLOR
                linestyle = "solid"
            else:
                label = "" if have_labelled_alt else "$H^C_{1a}\\;,H^C_{1b}$"
                have_labelled_alt = True
                color = figures.ALT_COLOR
                linestyle = "dashed"

            rows = rows.sort_values("d")
            (line,) = plt.plot(
                rows["d"].values,
                rows["power"].values,
                label=label,
                color=color,
                linestyle=linestyle,
            )

            if df != "null":
                label_x = 5.0 if df == 5.0 else 35.0
                labelLine(line, x=label_x, label=f"${df:.0f}$")

        plt.axhline(0.05, linestyle="--")
        plt.xlabel("d", **figures.squashed_label_params)
        plt.xlim(ds[0], ds[-1])
        plt.ylim(0, 1)
        plt.xticks([ds[0], ds[-1] // 2, ds[-1]])
        plt.yticks([0, 1])
        plt.legend(loc=(0.3, 0.15), ncol=3, **figures.squashed_legend_params)

        save_fig(runner)
        plt.close()

    runner.run(run_exp, plot)


def exp_other_tests(runner: ExperimentRunner) -> None:
    rng = PRNGKey(seed=80982342)
    d = 1
    ns = [50, 200]
    null_dist = Gaussian

    tests = [
        ShapiroWilks(
            props={
                "plot_args": {
                    "label": "S-W",
                    "linestyle": (0, (3, 1)),
                    "color": "black",
                },
            }
        ),
        Lilliefors(
            props={
                "plot_args": {
                    "label": "Lllfrs",
                    "linestyle": (0, (3, 1, 1, 1, 1, 1)),
                    "color": "black",
                },
            }
        ),
        Anderson(
            props={
                "plot_args": {
                    "label": "A-D",
                    "linestyle": "solid",
                    "color": "black",
                },
            }
        ),
        DAgostino(
            props={
                "plot_args": {
                    "label": "D’Agso",
                    "linestyle": "dotted",
                    "color": "black",
                },
            }
        ),
    ]
    ls: List[Tuple[Union[float, Literal["m"]], float, str]] = [
        ("m", 3.0, "C0"),
        (2.0, 7.0, "C1"),
    ]
    for i, (l, l_pos, color) in enumerate(ls):
        for bootstrap, linestyle in zip(Bootstrap, ["--", "-"]):
            l_str = "\\textrm{m}" if l == "m" else f"{l:.1f}"
            bs_str = {Bootstrap.WILD: "wld", Bootstrap.PARAMETRIC: "prm"}[bootstrap]
            tests.append(
                Ours(
                    lengthscale=l,
                    estimator=lambda l: KSDAnalyticEstimator(_k(l), null_dist),
                    test_statistic=lambda l: KSDStatistic(_k(l), null_dist),
                    d=d,
                    null_family=null_dist,
                    bootstrap=bootstrap,
                    props={
                        "plot_args": {
                            "linestyle": linestyle,
                            "label": f"$l={l_str}$, {bs_str}",
                            "color": color,
                        },
                        "l": l,
                        "l_pos": l_pos,
                    },
                )
            )

    dfs = jnp.arange(1.0, 10.0, 0.5).tolist()
    for n in ns:
        for df in dfs:
            for test in tests:
                other_dist = UnivariateT(df)
                rng, rng_input = jax.random.split(rng)
                config = Config(rng_input, other_dist, df, n, d, test)
                name = f"gpc_other_tests_{config.name}_df{df:.2f}"
                runner.queue(name, config)

    def plot(results: List[Tuple[Config, Result]]) -> None:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            sharey=True,
            figsize=(figures.FULL_WIDTH, 1.2 * figures.COMPACT_HEIGHT),
            gridspec_kw={"width_ratios": [0.6, 0.4]},
        )
        for ax, n in zip(axes, ns):
            for test in tests:
                test_rs = [(c, r) for c, r in results if c.test is test and c.n == n]
                vs, ps = zip(*[(c.x_val, r.power) for c, r in test_rs])
                ax.plot(vs, ps, **test.props["plot_args"])

            ax.set_title(f"$n={n}$")
            ax.set_xlim(left=1.0, right=max(dfs))
            ax.set_ylim(bottom=0.0, top=1.0)
            ax.set_xlabel("degrees of freedom, $\\nu$", **figures.squashed_label_params)
            ax.set_yticks([0, 1])
            ax.axhline(0.05, linestyle="--", color="black")

        axes[0].legend(ncol=2, **figures.squashed_legend_params)
        axes[0].set_ylabel("power", **figures.squashed_label_params)

        save_fig(runner)
        plt.close()

    runner.run(run_exp, plot)


def run_exp(config: Config) -> Result:
    rng = config.rng
    results = []
    for repeat in range(N_REPEATS):
        rng, rng_input = jax.random.split(rng)
        ys = config.other_distribution.sample(rng_input, config.n)

        # We need to recompute the median heuristic for each configuration, but we can
        # safely reuse it between repeats of the same configuration.
        use_median_heuristic_cache = repeat != 0
        rng, rng_input = jax.random.split(rng)
        results.append(config.test.run(rng_input, ys, use_median_heuristic_cache))

    power = sum(results) / len(results)
    if isinstance(config.test, Ours):
        median_heuristic_l = config.test.median_heuristic_l
    else:
        median_heuristic_l = None
    return Result(power, median_heuristic_l)


def save_fig(runner: ExperimentRunner) -> None:
    figures.save_fig(f"gpc_{runner.exp}")


if __name__ == "__main__":
    figures.configure_matplotlib()

    exp_funcs: Dict[str, Callable[[ExperimentRunner], None]] = {
        "increasing_n_null_ksd": exp_increasing_n_null_ksd,
        "increasing_n_alt_ksd": exp_increasing_n_alt_ksd,
        "increasing_n_null_mmd": exp_increasing_n_null_mmd,
        "mmd_vs_ksd": exp_mmd_vs_ksd,
        "increasing_d_wild": exp_increasing_d_wild,
        "increasing_d_parametric": exp_increasing_d_parametric,
        "other_tests": exp_other_tests,
    }
    runners.main(exp_funcs)
