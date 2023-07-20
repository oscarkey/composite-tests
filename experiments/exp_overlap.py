from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import runners
from jax import vmap
from jax.numpy import ndarray
from jax.random import KeyArray, PRNGKey
from runners import ExperimentRunner
from tqdm import tqdm

import figures
from composite_tests.bootstrapped_tests import TestStatistic
from composite_tests.distributions import SampleableDist
from composite_tests.distributions.gaussian import (
    Gaussian,
    GaussianFixedScaleMMDEstimator,
    gaussian_fixed_scale,
)
from composite_tests.estimators import Estimator
from composite_tests.kernels import GaussianKernel
from composite_tests.mmd import MMDStatistic

Mode = Literal["null", "alt"]


@dataclass
class Config:
    rng: KeyArray
    mode: Mode
    aim: Literal["bad_parameter", "good_parameter"]
    n: int
    n_bootstrap_samples: int
    opt_iterations: int
    learning_rate: float
    alt_std: float

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_"
            f"n{self.n}_"
            f"nbs{self.n_bootstrap_samples}_"
            f"opti{self.opt_iterations}_"
            f"lr{self.learning_rate:.3f}_"
            f"altstd{self.alt_std:.2f}"
        )


@dataclass
class Results:
    params: ndarray
    statistics: ndarray


def bad_optimizer(runner: ExperimentRunner) -> None:
    rng = PRNGKey(seed=90880913)
    enabled_modes: List[Mode] = ["null", "alt"]
    for mode in enabled_modes:
        rng, rng_input = jax.random.split(rng)
        config = Config(
            rng_input,
            mode,
            aim="good_parameter",
            n=400,
            n_bootstrap_samples=300,
            opt_iterations=30,
            learning_rate=0.5,
            alt_std=1.5,
        )
        runner.queue(f"overlap_{config.name}", config)
        rng, rng_input = jax.random.split(rng)
        config = Config(
            rng_input,
            mode,
            aim="bad_parameter",
            n=400,
            n_bootstrap_samples=300,
            opt_iterations=30,
            learning_rate=5.5,
            alt_std=1.5,
        )
        runner.queue(f"overlap_{config.name}", config)

    def plot(results: List[Tuple[Config, Results]]) -> None:
        figures.configure_matplotlib()
        fig, axes = plt.subplots(
            2, 2, figsize=(figures.FULL_WIDTH, figures.FULL_WIDTH * 0.5)
        )
        for row, aim in zip(axes, ["good_parameter", "bad_parameter"]):
            for config, result in results:
                if config.aim != aim:
                    continue
                row[0].set_ylabel(
                    {
                        "good_parameter": "good optimiser",
                        "bad_parameter": "poor optimiser",
                    }[aim]
                )
                hist_args = {"bins": 40, "alpha": 0.2, "density": True}
                label = {"null": "$H_0^C$", "alt": "$H_1^C$"}[config.mode]
                row[0].hist(result.statistics, **hist_args, label=label)
                row[1].hist(result.params, **hist_args, label=label)
                row[1].set_xlim(-0.7, 1.5)
        for ax in axes.flatten():
            ax.set_yticks([])
        axes[0, 1].legend()
        axes[1, 0].set_xlabel("test statistic")
        axes[1, 1].set_xlabel("parameter estimate")
        figures.save_fig("overlap_failure")

    runner.run(run_exp, plot)


def run_exp(config: Config) -> Results:
    rng = config.rng

    null_std = 1.0
    null_family = gaussian_fixed_scale(null_std)

    kernel = GaussianKernel(l=1.0)
    estimator = GaussianFixedScaleMMDEstimator(
        kernel,
        jnp.array(null_std),
        iterations=config.opt_iterations,
        learning_rate=config.learning_rate,
    )
    statistic = MMDStatistic(kernel, null_family)

    if config.mode == "null":
        print("Computing null")
        observations_dist = Gaussian(loc=0.5, scale=null_std)
    elif config.mode == "alt":
        print("Computing alt")
        observations_dist = Gaussian(loc=0.5, scale=config.alt_std)

    rng, *rng_inputs = jax.random.split(rng, num=config.n_bootstrap_samples + 1)
    return _sample_params_and_test_statistics(
        rng_inputs, estimator, statistic, observations_dist, batch_size=200, n=config.n
    )


def _sample_params_and_test_statistics(
    rngs: List[KeyArray],
    estimator: Estimator,
    statistic: TestStatistic,
    observations_dist: SampleableDist,
    n: int,
    batch_size: int,
) -> Results:
    batched_rngs = [rngs[i : i + batch_size] for i in range(0, len(rngs), batch_size)]
    params_and_statistics = [
        vmap(_sample_parameter_and_test_statistic, in_axes=(0, None, None, None, None))(
            jnp.array(batch), estimator, statistic, observations_dist, n
        )
        for batch in tqdm(batched_rngs)
    ]
    params, statistics = zip(*params_and_statistics)
    return Results(jnp.concatenate(params), jnp.concatenate(statistics))


def _sample_parameter_and_test_statistic(
    rng: KeyArray,
    estimator: Estimator,
    statistic: TestStatistic,
    observations_dist: SampleableDist,
    n: int,
) -> Tuple[ndarray, ndarray]:
    rng, rng_input = jax.random.split(rng)
    ys = observations_dist.sample(rng_input, n)
    rng, rng_input = jax.random.split(rng)
    theta_hat = estimator(rng_input, ys)
    rng, rng_input = jax.random.split(rng)
    return theta_hat[0], statistic(rng_input, theta_hat, ys)


if __name__ == "__main__":
    figures.configure_matplotlib()

    exp_funcs: Dict[str, Callable[[ExperimentRunner], None]] = {
        "bad_optimizer": bad_optimizer,
    }
    runners.main(exp_funcs)
