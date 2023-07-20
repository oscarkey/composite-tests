from typing import Callable, Tuple

import jax.numpy as jnp
import rdata
from jax.numpy import ndarray

UnnormalizeFunc = Callable[[ndarray], ndarray]


def load_galaxies() -> Tuple[ndarray, UnnormalizeFunc]:
    parsed = rdata.parser.parse_file("datasets/galaxies.rda")
    converted = rdata.conversion.convert(parsed, default_encoding="ASCII")
    unnormalized = jnp.array(converted["galaxies"]).reshape(-1, 1)

    location = unnormalized.mean()
    scale = 0.5 * unnormalized.std()
    normalized = (unnormalized - location) / scale

    def unnormalize(x: ndarray) -> ndarray:
        return x * scale + location

    return normalized, unnormalize
