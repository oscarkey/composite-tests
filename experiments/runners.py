"""Contains tools for caching experiment results for faster plotting."""

import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Callable, Dict, Generic, List, Tuple, TypeVar

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

T = TypeVar("T")
S = TypeVar("S")


class ExperimentRunner(ABC, Generic[T, S]):
    def __init__(self, exp: str) -> None:
        self._exp_configs: List[Tuple[str, T]] = []
        self.exp = exp

    def queue(self, name: str, config: T) -> None:
        self._exp_configs.append((name, config))

    @abstractmethod
    def run(
        self, run_exp: Callable[[T], S], plot: Callable[[List[Tuple[T, S]]], None]
    ) -> None:
        pass

    def _get_missing_configs(self) -> List[Tuple[str, T]]:
        missing_configs: List[Tuple[str, T]] = []
        for name, config in self._exp_configs:
            if not self._check_result_exists(name):
                print(f"Missing config {name}")
                missing_configs.append((name, config))
        return missing_configs

    def _load_results(self) -> List[Tuple[T, S]]:
        results: List[Tuple[T, S]] = []
        for name, config in self._exp_configs:
            if not self._check_result_exists(name):
                raise ValueError(f"Need to compute result {name}")
            results.append((config, self._load_result(name)))
        return results

    def _check_result_exists(self, name: str) -> bool:
        return os.path.exists(self._get_save_path(name))

    def _load_result(self, name: str) -> S:
        # Using np.load due to https://github.com/google/jax/issues/9700 .
        loaded = np.load(self._get_save_path(name), allow_pickle=True)
        # When a Python object is saved it gets put in a scalar array.
        # Here we try to detect when that has happened, and return the contained object
        # rather than the array.
        if loaded.shape == () and loaded.dtype == jnp.dtype("O"):
            return loaded.item()
        else:
            return loaded

    def _save_result(self, name: str, result: S) -> None:
        return jnp.save(self._get_save_path(name), result)

    def _get_save_path(self, name: str) -> str:
        return os.path.join("results", f"{name}.npy")


class LocalRunner(ExperimentRunner):
    def run(
        self, run_exp: Callable[[T], S], plot: Callable[[List[Tuple[T, S]]], None]
    ) -> None:
        for name, config in tqdm(self._get_missing_configs()):
            result = run_exp(config)
            print("Trying to save ", name)
            self._save_result(name, result)
        plot(self._load_results())


ExperimentFunc = Callable[[ExperimentRunner], None]


def main(exp_funcs: Dict[str, ExperimentFunc]) -> None:
    parser = ArgumentParser()
    exp_choices = list(exp_funcs.keys()) + ["all"]
    parser.add_argument("exp", choices=exp_choices)
    args = parser.parse_args()

    if args.exp == "all":
        exps = list(exp_funcs.keys())
    else:
        exps = [args.exp]

    for exp in exps:
        exp_funcs[exp](LocalRunner(exp))
