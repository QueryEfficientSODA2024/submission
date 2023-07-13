from typing import Callable
from functools import wraps
import numpy as np
from numpy.typing import NDArray

Entry = tuple[int, int]


def validate(gen_fn: Callable) -> Callable:
    """
    Decorator for matrix generation functions.
     Validates that the generated matrix has a unique Nash equilibrium.
    :param gen_fn: function that returns a matrix and a Nash equilibrium.
    :return: wrapped function.
    """

    @wraps(gen_fn)
    def _wrapped(*args, **kwargs):
        tries = 0
        max_tries = 100
        while tries < max_tries:
            a, (i_star, j_star) = gen_fn(*args, **kwargs)
            n = a.shape[0]
            val = a[i_star, j_star]
            try:
                if n > 1:
                    # check that we have strictly positive gaps
                    row_sorted = np.sort(a[i_star, :])
                    assert val < row_sorted[1]
                    col_sorted = np.sort(
                        a[:, j_star],
                    )
                    assert val > col_sorted[-2]
                    # check that we have a unique minimum in the row and maximum in the column
                    for rix in range(n):
                        row_mins = np.nonzero(a[rix, :] == np.min(a[rix, :]))[0]
                        for row_min_ix in row_mins:
                            if rix == i_star:
                                assert row_min_ix == j_star
                            else:
                                assert a[rix, row_min_ix] < np.max(a[:, row_min_ix])
            except AssertionError:
                tries += 1
                continue
            else:
                return a, (i_star, j_star)
        raise ValueError("Could not generate matrix")

    return _wrapped


@validate
def _random_floats(
    n: int, rng: np.random.Generator, centered: bool = True
) -> tuple[NDArray, Entry]:
    """
    Generate a random matrix with a unique Nash equilibrium, where most entries are drawn from a uniform distribution.
    :param n: size of the matrix.
    :param rng: np.random.Generator instance.
    :param centered: if true, the Nash value is drawn from [1/3, 2/3].
     Otherwise, it is drawn from [0, 1/3] or [2/3, 1].
    :return: tuple of the matrix and its Nash equilibrium.
    """
    a = rng.uniform(size=(n, n))
    if centered:
        val = rng.uniform(low=1 / 3, high=2 / 3)
    else:
        if rng.uniform() < 0.5:
            val = rng.uniform(low=0, high=1 / 3)
        else:
            val = rng.uniform(low=2 / 3, high=1)

    i_star = rng.integers(n)
    j_star = rng.integers(n)
    a[i_star, :] = rng.uniform(low=val, size=n)
    a[:, j_star] = rng.uniform(high=val, size=n)
    a[i_star, j_star] = val
    # put on readable scale
    return a * 10, (i_star, j_star)


def random_centered(n: int, rng: np.random.Generator) -> tuple[NDArray, Entry]:
    """
    Generate a random matrix with a unique Nash equilibrium, where most entries are drawn from a uniform distribution.
    The Nash equilibrium value is drawn from [1/3, 2/3].
    :param n: size of the matrix.
    :param rng: np.random.Generator instance.
    :return: tuple of the matrix and its Nash equilibrium.
    """
    return _random_floats(n, rng, centered=True)


def random_skewed(n: int, rng: np.random.Generator) -> tuple[NDArray, Entry]:
    """
    Generate a random matrix with a unique Nash equilibrium, where most entries are drawn from a uniform distribution.
    The Nash equilibrium value is drawn from [0, 1/3] or [2/3, 1].
    :param n: size of the matrix.
    :param rng: np.random.Generator instance.
    :return: tuple of the matrix and its Nash equilibrium.
    """
    return _random_floats(n, rng, centered=False)


@validate
def random_binary(n: int, rng: np.random.Generator) -> tuple[NDArray, Entry]:
    """
    Generate a random binary matrix with a unique Nash equilibrium.
    The game value is 1/2, with the row and column set to 1s and 0s.
    :param n: size of the matrix.
    :param rng: np.random.Generator instance.
    :return: tuple of the matrix and its Nash equilibrium.
    """
    a = rng.integers(low=0, high=2, size=(n, n)).astype(float)
    i_star = rng.integers(n)
    j_star = rng.integers(n)

    a[i_star, :] = 1.0
    a[:, j_star] = 0.0
    a[i_star, j_star] = 0.5

    return a, (i_star, j_star)


@validate
def random_curved(n: int, rng: np.random.Generator) -> tuple[NDArray, Entry]:
    """
    Generate a random matrix with a unique Nash equilibrium.
    Entries are set proportional to 5i+j and then are perturbed.
    :param n: size of the matrix.
    :param rng: np.random.Generator instance.
    :return: tuple of the matrix and its Nash equilibrium.
    """
    r = np.arange(n)
    a = 5.0 * r[:, None] + r[None, :]
    a /= np.max(a[:])
    a += rng.uniform(high=0.3, size=(n, n))

    # pick a row in the lower half (higher row val)
    i_star = rng.integers(low=n // 2, high=n)
    j_star = np.argmin(a[i_star, :], keepdims=True)[0]
    val = a[i_star, j_star]
    # scale rest of column to be a little lower than val
    a[:, j_star] *= rng.uniform(low=0.75, high=0.99) * val / np.max(a[:, j_star])
    a[i_star, j_star] = val
    # put on readable scale
    return a * 10, (i_star, j_star)
