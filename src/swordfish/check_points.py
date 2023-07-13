from typing import Optional, Iterable
from numpy.typing import NDArray

Entry = tuple[int, int]


def check_psne(
    pt: Entry, a: NDArray, queries: NDArray, rows: Iterable, cols: Iterable
) -> tuple[Optional[Entry], NDArray]:
    """
    Checks to see if pt is a PSNE of A, assuming that we only need to check the rows and columns in rows and cols.
    :param pt: Entry to check.
    :param a: NDArray representing the game matrix.
    :param queries: NDArray representing the queries made so far.
    :param cols: Columns to check.
    :param rows: Rows to check.
    :return: (pt, queries) if pt is a PSNE, (None, queries) otherwise.
    """
    val = queries[pt]
    for i in rows:
        comp_pt = a[i, pt[1]]
        queries[i, pt[1]] = comp_pt
        if comp_pt >= val and i != pt[0]:
            return None, queries
    for j in cols:
        comp_pt = a[pt[0], j]
        queries[pt[0], j] = comp_pt
        if comp_pt <= val and j != pt[1]:
            return None, queries
    return pt, queries


def compare_pts(
    pt1: Entry, pt2: Entry, b: NDArray, queries: NDArray
) -> tuple[Optional[Entry], NDArray]:
    """
    Compares two points, pt1 and pt2, and returns one of them it is possibly the PSNE of B.
    If neither is possibly the PSNE, returns None.
    :param pt1: Entry to compare.
    :param pt2: Entry to compare.
    :param b: NDArray representing the game matrix.
    :param queries: NDArray representing the queries made so far.
    :return: tuple of [pt1, pt2, or None], along with the updated queries.
    """
    val_1, val_2 = queries[pt1], queries[pt2]
    # swap to ensure pt1 has the lower value
    if val_1 > val_2:
        return compare_pts(pt2, pt1, b, queries)
    i1, j1 = pt1
    i2, j2 = pt2
    if i1 == i2:
        nash = pt1 if val_1 < val_2 else pt2
    elif j1 == j2:
        nash = pt1 if val_1 > val_2 else pt2
    else:
        pivot = b[i2, j1]
        queries[i2, j1] = pivot
        if val_1 > pivot:
            nash = pt1
        elif val_2 < pivot:
            nash = pt2
        else:
            nash = None
    return nash, queries
