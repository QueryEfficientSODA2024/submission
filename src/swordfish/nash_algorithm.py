from typing import Optional

import numpy as np
from numpy.typing import NDArray

Entry = tuple[int, int]


class NashAlgorithm:
    """
    Base class for computing the unique pure-strategy Nash equilibrium in a zero-sum matrix game.
    """

    def __init__(self) -> None:
        self.a: NDArray = np.array([])
        self.n: Optional[int] = None
        self.queries: NDArray = np.array([])
        self.costs: dict = dict()

    def solve(self, a: NDArray) -> Entry:
        """
        Solves the game defined by A.
        :param a: NDArray of shape (n, n) representing the matrix game.
        :return: tuple of the Nash equilibrium.
        """
        raise NotImplementedError

    def _add_cost(self, value: int, type_: str) -> None:
        """
        Adds a cost to the dict of total algorithm costs.
        :param value: Cost amount.
        :param type_: Str representing the type of cost.
        """
        self.costs[type_] = self.costs.get(type_, 0) + value

    def cost(self, unique: bool = False) -> int:
        """
        Returns the cost of the algorithm.
        :param unique: If True, returns the number of unique queries actually made.
         If False, returns the number of query attempts;
          i.e., the number of times the algorithm attempted to query an entry.
        :return: Cost of the algorithm.
        """
        if unique:
            return np.count_nonzero(~np.isnan(self.queries))
        else:
            return sum(self.costs.values())

    def normalized_cost(self, unique: bool = False) -> float:
        """
        Returns the cost of the algorithm relative to the bound.
        :param unique: If True, returns the true number of queries of the algorithm divided by n^2.
         If False, returns the number of query attempts divided by n^2.
        :return: Normalized cost of the algorithm.
        """
        if self.n is None:
            raise ValueError("n must be set before calling normalized_cost.")
        return self.cost(unique=unique) / (self.n**2)
