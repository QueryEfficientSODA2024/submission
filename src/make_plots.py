#!/usr/bin/env python3

from typing import Iterable

import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from swordfish.swordfish import Swordfish
from swordfish.problems import (
    random_centered,
    random_skewed,
    random_binary,
    random_curved,
)


from swordfish.base_logger import logger
from swordfish.plotting import plot_stats


def recurrence_bound(n: int) -> float:
    # slightly tighter bound with f(2)==3
    return 1 if n == 1 else n * (3 / 2 * np.log2(n))


def c_log2_n(c: float):
    """With f(n) = c * log2(n), returns (str, fn)."""

    def f(n_: int) -> float:
        return c * np.log2(n_)

    return f"{c}log_n", f


def run_experiments(
    rng: np.random.Generator,
    exp_range: Iterable,
    n_trials: int,
    algorithms: Iterable,
    problem_types: Iterable,
) -> pd.DataFrame:
    """
    Run experiments for a range of n values and a number of trials.
    :param rng: np.random.Generator instance.
    :param exp_range: iterable of exponents (n=2^exp).
    :param n_trials: number of trials per n.
    :param algorithms: list of algorithms to run.
    :param problem_types: list of problem types (i.e. callable functions) to run.
    :return: DataFrame of results.
    """
    results = []
    tqdm_options = {
        "leave": False,
        "ncols": 80,
        "colour": "white",
        "disable": False,
    }
    with logging_redirect_tqdm():
        pbar = tqdm(exp_range, desc="Exponent", **tqdm_options)
        for exp in pbar:
            pbar.set_postfix({"n": f"2^{exp}"})
            n = 2**exp
            for k in trange(n_trials, desc="Trial", **tqdm_options):
                for problem_type in problem_types:
                    a, nash = problem_type(n, rng=rng)
                    for algorithm in algorithms:
                        nash_guess = algorithm.solve(a)
                        correct = nash == nash_guess
                        if not correct:
                            logger.warning(
                                f"{str(algorithm)}|n=2^{exp}|{problem_type.__name__}|{k=} is incorrect."
                            )
                        for unique in [True, False]:
                            cost = algorithm.cost(unique=unique)
                            results.append(
                                {
                                    "n": n,
                                    "cost_type": "Unique Queries"
                                    if unique
                                    else "Attempted Queries",
                                    "cost": cost,
                                    "normalized_cost": cost / recurrence_bound(n),
                                    "correct": correct,
                                    "algorithm": str(algorithm),
                                    "problem_type": problem_type.__name__,
                                }
                            )
    df = pd.DataFrame(results)
    # double-check correctness of algorithms
    df_g = (
        df.groupby(["n", "algorithm", "problem_type"])
        .agg({"n": "first", "algorithm": "first", "correct": "mean"})
        .reset_index(drop=True)
    )
    inc_df = df_g[df_g["correct"] < 1]
    if len(inc_df):
        logger.warning("Some algorithms are not 100% correct:")
        for alg, problem_type, correct, n in zip(
            inc_df["algorithm"],
            inc_df["problem_type"],
            inc_df["correct"],
            inc_df["n"],
        ):
            logger.warning(f"{alg}: {correct:.1%} correct for n={n}|{problem_type}")
    return df


if __name__ == "__main__":
    rng_ = default_rng(sum(map(ord, "SODA2024")))
    # experiment 1: each optimization turned on by itself
    exp_range_ = np.arange(3, 11)
    n_trials_ = 200
    algorithms_ = (
        Swordfish(),
        Swordfish(quick_move=True),
        Swordfish(quick_mark=True),
        Swordfish(recycle=True),
        Swordfish(brute_force=c_log2_n(2)),
    )
    logger.info("Running experiment 1.")
    df1 = run_experiments(rng_, exp_range_, n_trials_, algorithms_, (random_centered,))
    df1.to_csv(f"./swordfish_all_opts_comparison.csv", index=False)
    plot_stats(df1, plot_type=1)

    # experiment 2: base vs fast, all 4 types
    n_trials_ = 500
    algorithms_ = (Swordfish(), Swordfish(fast=True))
    problem_types_ = (
        random_centered,
        random_skewed,
        random_binary,
        random_curved,
    )
    logger.info("Running experiment 2.")
    df2 = run_experiments(
        rng_,
        exp_range_,
        n_trials_,
        algorithms_,
        problem_types_,
    )
    df2.to_csv("./swordfish_all_types_comparison.csv", index=False)
    plot_stats(df2, plot_type=2)
