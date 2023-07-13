from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray

from swordfish.nash_algorithm import NashAlgorithm, Entry
from swordfish.sorting import diag_match, sort2
from swordfish.check_points import compare_pts

Subproblem = tuple[slice, slice, bool]


class RecursiveSortAlgorithm(NashAlgorithm):
    """
    Recursive algorithm for finding PSNE of a matrix game.
    Uses sorting of subproblem diagonals to find a PSNE.
    """

    def __init__(
        self,
        brute_force: int | tuple[str, Callable] = 2,
        recycle: bool = False,
    ) -> None:
        """
        Constructor for RecursiveSortAlgorithm.
        :param brute_force: One of int or tuple[str, Callable].
         If int, then the algorithm will use brute force for subproblems of size <= brute_force.
         If tuple[str fn_name, Callable fn], then the algorithm will use fn to determine
            whether to use brute force for a given subproblem.
        :param recycle: Whether to recycle path queries from subproblems.
        """
        super().__init__()
        self.warm_diag = True
        self.flags = {
            "recycle": recycle,
        }
        if isinstance(brute_force, int):
            self.brute_force_fn = lambda n: n <= brute_force
            if brute_force > 2:
                self.flags[f"brute_force[{brute_force}]"] = True
        else:
            brute_force_name, self.brute_force_fn = brute_force
            self.flags[f"brute_force[{brute_force_name}]"] = True

    def __str__(self) -> None:
        raise NotImplementedError

    def solve(self, a: NDArray) -> Entry:
        """
        Solves a matrix game using the recursive algorithm.
        :param a: NDArray representing the matrix game.
        :return: tuple of the form (row, col) representing the PSNE of A.
        """
        self.costs = dict()  # need to reset costs
        self.a = a
        self.n = a.shape[0]
        assert a.shape[0] == a.shape[1], "a must be square"
        assert np.log2(self.n) % 1 == 0, "n must be a power of 2"

        self.queries = np.nan * np.zeros_like(a)
        self.nullified = np.zeros((self.n, self.n), dtype=bool)

        (nash, self.queries, self.nullified) = self._solve(
            a, self.queries, self.nullified
        )
        return nash

    def _solve_brute_force(
        self, b: NDArray, queries: NDArray, nullified: NDArray
    ) -> tuple[Optional[Entry], NDArray, NDArray]:
        """
        Solves a subproblem using brute force, i.e. by querying all non-nullified points.
        :param b: NDArray representing the subproblem.
        :param queries: NDArray representing queries made.
        :param nullified: NDArray representing the nullified entries.
        :return: tuple of the form (nash, queries, nullified) representing the PSNE of b.
            If there is no PSNE, then nash is None.
        """
        n = b.shape[0]
        if n == 1:
            # this branch will only be reached if original problem is 1x1
            nash = (0, 0)
        elif n == 2:
            self._add_cost(1, "compare")  # only cost, as we already paid for diag
            nash, queries = compare_pts((0, 0), (1, 1), b, queries)
            if not nash:
                nash = (1, 0)
        else:
            # check all non-nullified points
            rows, cols = np.where(~nullified)
            potential_nash = [(r, c) for r, c in zip(rows, cols)]
            # we have at most |pts| - 1 compares, unless |pts| == 1
            # slightly pessimistic, as some compares may return None
            self._add_cost(max(len(potential_nash) - 1, 0), "bf_compare")
            nash = None
            for pt in potential_nash:
                self._add_cost(1, "bf_query")
                queries[pt] = b[pt]
                if nash is None:
                    nash = pt
                else:
                    nash, queries = compare_pts(nash, pt, b, queries)
        return nash, queries, nullified

    def _solve(
        self,
        b: NDArray,
        queries: NDArray,
        nullified: NDArray,
        depth: int = 0,
        warm_diag: bool = False,
    ) -> tuple[Optional[Entry], NDArray, NDArray]:
        """
        Solves a subproblem using the recursive algorithm.
        :param b: NDArray representing the subproblem.
        :param queries: NDArray representing queries made.
        :param nullified: NDArray representing the nullified entries.
        :param depth: int representing the depth of the subproblem.
        :param warm_diag: bool representing whether the diagonal has already been queried.
        :return: tuple of the form (nash, queries, nullified) representing the PSNE of b.
            If there is no PSNE, then nash is None.
        """
        n = b.shape[0]
        if warm_diag:
            row_ix, col_ix = np.arange(n), np.arange(n)
        else:
            # when permuting, we make copies to not worry about numpy views
            if self.flags["recycle"]:
                match_ix, diag_query_ix = diag_match(queries)
            else:
                match_ix, diag_query_ix = np.arange(n), np.arange(n)
            # we are paying for the diagonal in attempted queries regardless
            self._add_cost(n, "diag")
            matched_diag = np.diag(b[match_ix])
            diag_sort_ix = np.argsort(matched_diag)
            row_ix = match_ix[diag_sort_ix]
            col_ix = diag_sort_ix

            # sort the required subproblem arrays
            b = sort2(b, row_ix, col_ix)
            queries = sort2(queries, row_ix, col_ix)
            nullified = sort2(nullified, row_ix, col_ix)
            # overwrite the whole diagonal, we already paid for it
            queries[np.diag_indices(n)] = np.diag(b)

        # brute force small subproblems
        if (n <= 2) or (~nullified).sum() <= self.brute_force_fn(n):
            nash, queries, nullified = self._solve_brute_force(b, queries, nullified)
        else:
            # nullify the upper-right quadrant
            nullified |= ~np.tri(n, dtype=bool)
            # generate subproblems (at most 3, for n^1.58 alg, at most 2 for Swordfish)
            subproblems, queries, nullified = self._generate_subproblems(
                b, queries, nullified
            )
            nash = None
            for subproblem_ix, (i_slices, j_slices, can_warm) in enumerate(subproblems):
                slices = (i_slices, j_slices)
                nash_sub, queries[slices], nullified[slices] = self._solve(
                    b[slices],
                    queries[slices],
                    nullified[slices],
                    depth=depth + 1,
                    warm_diag=self.warm_diag and can_warm,
                )
                if nash_sub is not None:
                    # shift subproblem nash index to be in terms of b
                    nash_sub = (
                        nash_sub[0] + slices[0].start,
                        nash_sub[1] + slices[1].start,
                    )
                    # compare to current best point
                    if nash is None:
                        nash = nash_sub
                    else:
                        self._add_cost(1, "compare")
                        nash, queries = compare_pts(nash, nash_sub, b, queries)
        # unsort the subproblem arrays to be back in terms of the parent problem
        inv_row_ix = np.argsort(row_ix)
        inv_col_ix = np.argsort(col_ix)
        return (
            (row_ix[nash[0]], col_ix[nash[1]]) if nash is not None else None,
            sort2(queries, inv_row_ix, inv_col_ix),
            sort2(nullified, inv_row_ix, inv_col_ix),
        )

    def _generate_subproblems(
        self, b: NDArray, queries: NDArray, nullified: NDArray
    ) -> tuple[list[Subproblem], NDArray, NDArray]:
        """
        Generates subproblems for the recursive algorithm.
        :param b: NDArray representing the subproblem.
        :param queries: NDArray representing queries made.
        :param nullified: NDArray representing the nullified entries.
        :return: tuple of the form (subproblems, queries, nullified) representing the
            subproblems of b. Each subproblem is a tuple of the form
            (i_slices, j_slices, can_warm) where i_slices and j_slices are tuples of
             extents and can_warm is a bool representing whether the diagonal can be
             assumed to be already queried.
        """
        raise NotImplementedError
