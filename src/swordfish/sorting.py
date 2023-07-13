import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy import sparse


def diag_match(queries: NDArray) -> tuple[NDArray, NDArray]:
    """
    Permutes the queries to maximize the number of already-queried diagonal elements.
    :param queries: NDArray representing the queries made.
    :return: tuple of the form (row_p, diag_indices) where row_p is the permutation
        that maximizes the number of already-queried diagonal elements and diag_indices
        is the set of indices of the diagonal elements still to be queried
    """
    n = queries.shape[0]
    g = sparse.csr_matrix(~np.isnan(queries))
    # returns permutation with -1 for unmatched rows
    row_p = maximum_bipartite_matching(g)
    diag_indices = np.flatnonzero(row_p == -1)
    # replace -1s with remainder of permutation
    row_p[diag_indices] = np.setdiff1d(np.arange(n), row_p[row_p != -1])
    return row_p, diag_indices


def diag_sort(a: NDArray) -> NDArray:
    """Sorts rows/cols of A according to diagonal entries. Returns sort index."""
    return np.argsort(np.diag(a))


def sort2(a: NDArray, row_ix: NDArray, col_ix: NDArray) -> NDArray:
    """Sorts rows/cols of A according to row_ix and col_ix. Returns sorted A."""
    return a[row_ix][:, col_ix]
