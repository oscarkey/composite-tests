# Composite Goodness-of-Fit Tests with Kernels

Composite goodness-of-fit tests answer the question "is it likely my data was sampled from any distribution in a given parametric family?"

In maths: given a parametric family of distributions $`\mathbb{P}_\theta`$ for $`\theta \in \Theta`$, and some data $`\{x_i\}_{i=1}^n \sim \mathbb{Q}`$, the test compares the null hypothesis $`\exists \theta \in \Theta`$ such that $`\mathbb{P}_\theta = \mathbb{Q}`$, against the alternative that $`\nexists \theta \in \Theta`$ such that $`\mathbb{P}_\theta = \mathbb{Q}`$.

This repository allows you to reproduce the experiments in our paper:
[Composite Goodness-of-fit Tests with Kernels](https://arxiv.org/abs/2111.10275); Oscar Key*, Arthur Gretton, François-Xavier Briol, Tamara Fernandez*

It also allows you to implement these tests for your own models or data.

If you have any feedback or questions, feel free to open an issue or [email me](https://oscarkey.github.io).


## Using this repository
You can [reproduce the results from the paper](#to-reproduce-the-results-from-the-paper), [use the tests in your own project](#to-use-the-tests-in-your-own-project), or [develop this repository](#to-develop-the-code) (I love receiving pull requests!).

The tests are implemented in JAX and will use a GPU if available.

### To reproduce the results from the paper
Set up the environment:
- Install dependencies: Python >=3.10 (I recommend using [pyenv](https://github.com/pyenv/pyenv)), [Poetry](https://python-poetry.org/)
- Install the environment including the experiments: `poetry install --with experiments`
- Activate the environment: `poetry shell`

The experiments are implemented in the scripts `experiments/exp_*.py`, and [reproduce_experiments.md](reproduce_experiments.md) gives the command to reproduce each figure.

### To use the tests in your own project
#### Install the package
`pip install git+https://github.com/oscarkey/composite-tests`

#### Define your model, by implementing either
- `composite_tests.distributions.UnnormalizedDist` if your model has a tractable unnormalized density function
- `composite_tests.distributions.SampleableDist` if your model has no tractable unnormalized density function, but can be sampled from (e.g. a simulator)

#### Define an estimator for your model
Implement `composite_tests.estimators.Estimator`.
Our theoretical guarantees assume that the estimator is of the form $`\arg \min_\theta \text{MMD}(\mathbb{P}_\theta, \{x_1\}_{i=1}^n)`$ or $`\arg \min_\theta \text{KSD}(\mathbb{P}_\theta, \{x_1\}_{i=1}^n)`$.
It's up to you how to solve the argmin, but we include tools for two options:
- A ready-made KSD estimator for exponential family distributions, `composite_tests.ksd.KSDAnalyticEstimator`
- We interface with gradient-based optimizers provided by [Optax](https://github.com/deepmind/optax), see `composite_tests.optimizers`. The simulator implemented in `composite_tests.distributions.toggle_switch` is an example of how to use this.

TODO: include ready-made gradient-based estimators for the MMD and KSD

#### Run the test
```python
from jax.random import PRNGKey, split
from composite_tests.kernels import GaussianKernel
from composite_tests.bootstrapped_tests import wild_bootstrap_test, parametric_bootstrap_test

kernel = GaussianKernel(l=1.0)
dist_family = MyModel
ys = your data
estimator = MyEstimator(kernel) # (e.g. KSDAnalyticEstimator(kernel, dist_family))
statistic = KSDStatistic(kernel, dist_family) # or MMDStatistic(kernel, dist_family)

rng = PRNGKey(seed=38123)
rng, rng_input = split(rng)
# If you want to use the parametric bootstrap.
test_result = parametric_bootstrap_test(
    rng_input, ys, estimator, dist_family, statistic, n_bootstrap_samples=400,
)
# Or, if you want to use the wild bootstrap.
test_result = wild_bootstrap_test(
    rng_input, ys, estimator, statistic, n_bootstrap_samples=500
)
```
See the paper for a discussion about the trade off between the wild and parametric bootstraps.

Our theoretical guarantees assume that you use the same kernel for the estimator and the test statistic, and use the KSD/MMD for both the test statistic and the estimator.
TODO: update the API to enforce this.


### To develop the code
- Install following ["To reproduce the results from the paper"](#to-reproduce-the-results-from-the-paper) above
- To format: `black` and `isort`
- To run the unit tests: `pytest`
- To run the type checker:
    - `mypy --ignore-missing-imports -p composite_tests`
    - `mypy --ignore-missing-imports experiments`


## Citation
If you use this code for your own projects, please consider citing our paper:
```
@misc{key2023composite,
      title={Composite Goodness-of-fit Tests with Kernels},
      author={Oscar Key and Arthur Gretton and François-Xavier Briol and Tamara Fernandez},
      year={2023},
      eprint={2111.10275},
      archivePrefix={arXiv},
}
```

## License
This repository is released under the MIT license (with the exception of `datasets`).
See `LICENSE.txt`.
