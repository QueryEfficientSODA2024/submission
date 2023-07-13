from typing import Callable
import numpy as np
from numpy.typing import NDArray

from swordfish.recursive_sort_algorithm import RecursiveSortAlgorithm, Subproblem


class Swordfish(RecursiveSortAlgorithm):
    """
    Implementation of the Swordfish algorithm.
    """

    def __init__(
        self,
        quick_move: bool = False,
        quick_mark: bool = False,
        recycle: bool = False,
        fast: bool = False,
        brute_force: int | tuple[str, Callable] = 2,
    ):
        """
        Constructor for Swordfish algorithm.
        :param quick_move: If true, moves without querying, if possible.
        :param quick_mark: If true, marks entries as nullified before recursion.
        :param recycle: If true, recycles path queries into diagonal queries.
        :param fast: If true, enables all optimization flags.
        :param brute_force: Brute force algorithm to use for small subproblems.
          Defaults to brute forcing 2x2 subproblems.
        """
        if fast:
            quick_move = True
            quick_mark = True
            recycle = True
            brute_force = ("2log_n", lambda n: 2 * np.log2(n))
        super().__init__(brute_force=brute_force, recycle=recycle)
        self.fast = fast
        self.flags = {
            **self.flags,
            "quick_move": quick_move,
            "quick_mark": quick_mark,
        }

    def __str__(self):
        if self.fast:
            return "Swordfish(fast)"
        flags = ",".join(k for k in self.flags if self.flags[k])
        return "Swordfish" + (f"({flags})" if flags else "")

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
        if self.flags["quick_mark"]:
            marks = self.mark_full(queries)
            nullified |= marks

        # main Swordfish algorithm
        n = b.shape[0]
        n_half = n // 2
        # diagonal has already been queried
        diag = np.diag(queries)
        r, c = (n - 1, 0)
        cost_path = 0
        while (r >= n_half) and (c < n_half):
            if self.flags["quick_move"]:
                try:
                    # try to move early by passing in current row and col
                    move_up, move_right = self.next_move(
                        n // 2,
                        nullified_row=nullified[r, c:],
                        nullified_col=nullified[: (r + 1), c],
                    )
                except AssertionError:
                    pass
                else:
                    r -= move_up
                    c += move_right
                    continue
            # query current element
            cost_path += 1
            b_rc = b[r, c]
            queries[r, c] = b_rc

            # get diagonal for current square of interest
            diag_sl = slice(c, r + 1)
            diag_size = r + 1 - c
            current_diag: NDArray = diag[diag_sl]
            # Use diagonal as row mins and col maxs for nullification
            col_maxs = np.zeros(n - c)
            col_maxs[:diag_size] = current_diag
            row_mins = np.zeros(r + 1)
            row_mins[-diag_size:] = current_diag
            # make sure to nullify entries above the diagonal (will be used by next_move)
            # setting to True now is faster than initializing to +/-inf earlier
            nullified_row = self.mark_nullified(row_mins=b_rc, col_maxs=col_maxs)
            nullified_row[diag_size:] = True
            nullified_col = self.mark_nullified(row_mins=row_mins, col_maxs=b_rc)
            nullified_col[:-diag_size] = True
            # mark entries as nullified
            nullified[r, c:] |= nullified_row
            nullified[: (r + 1), c] |= nullified_col

            if self.flags["quick_move"]:
                # use total nullified history to move, not only the just-marked entries
                nullified_row = nullified[r, c:]
                nullified_col = nullified[: (r + 1), c]

            move_up, move_right = self.next_move(
                n // 2,
                nullified_row,
                nullified_col,
            )
            assert move_up or move_right, "No valid moves"
            r -= move_up
            c += move_right
        self._add_cost(cost_path, "path")

        # recurse on lower-left quadrant
        first_half = slice(0, n_half)
        second_half = slice(n_half, n)
        # recurse on upper-left or bottom-right, if necessary
        up_exit, right_exit = r < n_half, c == n_half
        subproblems = []
        if not (up_exit and right_exit):
            if up_exit:
                subproblems.append((first_half, first_half, True))
            else:
                subproblems.append((second_half, second_half, True))
        subproblems.append(
            (second_half, first_half, False),
        )
        return subproblems, queries, nullified

    @staticmethod
    def mark_full(queries: NDArray) -> NDArray:
        """
        Marks entries as nullified if they are not (or can't be) the unique row min or unique col max.
        Full version used with the quick_mark flag.
        :param queries: NDArray representing the queries made.
        :return: NDArray representing entries to be nullified.
        """
        # an entry needs to be the unique row min and unique col max
        row_mins = np.nanmin(queries, axis=1)[:, None]
        col_maxs = np.nanmax(queries, axis=0)[None, :]
        # broadcasts across rows and cols
        nullified = row_mins <= col_maxs
        is_row_min = queries == row_mins
        is_col_max = queries == col_maxs
        # these also broadcast across rows and cols
        unique_row_min = is_row_min & (np.sum(is_row_min, axis=1) == 1)[:, None]
        unique_col_max = is_col_max & (np.sum(is_col_max, axis=0) == 1)[None, :]
        return nullified & ~(unique_row_min & unique_col_max)

    @staticmethod
    def mark_nullified(row_mins: NDArray, col_maxs: NDArray) -> NDArray:
        """
        Marks entries as nullified if they are not (or can't be) the unique row min or unique col max.
        Quicker version used in the core Swordfish movement check.
        :param row_mins: (m, 1) NDArray representing the row mins.
        :param col_maxs: (1, n) NDArray representing the col maxs.
        :return: (m, n) NDArray representing entries to be nullified.
        """
        return row_mins <= col_maxs

    @staticmethod
    def next_move(
        threshold: int,
        nullified_row: NDArray,
        nullified_col: NDArray,
    ) -> tuple[bool, bool]:
        """
        Computes the next move to make in the Swordfish algorithm.
        :param threshold: number of entries needed for movement. Will always be n / 2.
        :param nullified_row: NDArray representing the nullified entries in the current row.
        :param nullified_col: NDArray representing the nullified entries in the current column.
        :return: tuple of booleans representing whether to move up and right, respectively.
        """
        move_up, move_right = False, False
        if all(nullified_row[-threshold:]):
            move_up = True
            if all(nullified_col[:-1]):
                move_right = True
        if all(nullified_col[:threshold]):
            move_right = True
            if all(nullified_row[1:]):
                move_up = True
        assert move_up or move_right, "No valid moves"
        return move_up, move_right
