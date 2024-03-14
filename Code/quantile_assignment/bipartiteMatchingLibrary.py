from typing import Tuple, List

import numpy as np

from .hopcroftKarp import HopcroftKarp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from scipy.optimize import linear_sum_assignment

Matches = Tuple[int, List[np.ndarray]]


# NOTE: Hopcroft_karp doesn't match any pairs that have affinity above q value.
# The cost is calculated from min of unassigned vertices
def hopcroft_karp_custom(cost_matrix) -> Matches:
    cost_matrix = 1 - cost_matrix

    matches = HopcroftKarp(cost_matrix).match()

    cost = min(cost_matrix.shape) - len(matches[0])

    return cost, matches


# NOTE: Hopcroft_karp doesn't match any pairs that have affinity above q value.
# The cost is calculated from min of unassigned vertices
def hopcroft_karp_scipy(cost_matrix) -> Matches:
    cost_matrix = 1 - cost_matrix

    graph = csr_matrix(cost_matrix)
    perm = maximum_bipartite_matching(graph, perm_type="column")

    row_ind = np.where(perm > -1)[0]
    col_ind = perm[row_ind]

    cost = min(cost_matrix.shape) - len(row_ind)

    return cost, [row_ind, col_ind]


# NOTE: Hungarian algorithm returns matches even if they are below q value.
def hungarian(cost_matrix) -> Matches:
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=False)
    cost = cost_matrix[row_ind, col_ind].sum()
    return cost, [row_ind, col_ind]


# NOTE: Doesn't return 0 cost matches which would be invalid
def hungarian_cost_sensitive(cost_matrix) -> Matches:
    matches = linear_sum_assignment(cost_matrix, maximize=True)

    nonzero_matches = tuple(np.argwhere(cost_matrix[tuple(matches)] > 0).T)

    row_ind, col_ind = matches[0][nonzero_matches], matches[1][nonzero_matches]

    cost = min(cost_matrix.shape) - len(row_ind)
    return cost, [row_ind, col_ind]


matching_lib = {
    "hopcroft_custom": hopcroft_karp_custom,
    "hopcroft_scipy": hopcroft_karp_scipy,
    "hungarian_scipy": hungarian,
    "hungarian_cost": hungarian_cost_sensitive
}