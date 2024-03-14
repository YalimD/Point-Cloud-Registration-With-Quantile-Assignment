import copy
import decimal
import logging
import math
import random

from numba import jit

from Code.quantile_assignment.bipartiteMatchingLibrary import *


@jit(nopython=True)
def calculate_affinity_matrix(affinity_matrix,
                              features_a,
                              features_b,
                              penalty_coefficient):
    for i, feature_a in enumerate(features_a):
        for j, feature_b in enumerate(features_b):
            affinity_matrix[i, j] = np.exp(-penalty_coefficient
                                           * np.power(np.linalg.norm(feature_a - feature_b), 1))


def calculate_affinity(features_a,
                       features_b,
                       penalty_coefficient,
                       save_affinity_matrix):
    affinity_shape = (features_a.shape[0], features_b.shape[0])
    logging.info(f"Expected affinity size: {affinity_shape}")

    affinity_matrix = np.zeros(affinity_shape)

    calculate_affinity_matrix(affinity_matrix,
                              features_a,
                              features_b,
                              penalty_coefficient)

    if save_affinity_matrix:
        np.savetxt("../affinity.csv", affinity_matrix, delimiter=",", fmt="%1.6e")

    return affinity_matrix


def find_k_alpha(affinity_shape,
                 alpha,
                 reformulate=True):
    min_N, max_N = np.sort(affinity_shape)

    # Reformulate the alpha as "ratio of expected matches for smaller point cloud"
    if reformulate:
        k_alpha = max(1, math.ceil((1 - alpha) * min_N))
    # As in paper; ratio of unmatched and bad matches over all possibilities
    else:
        alpha_min = (min_N - 1) / min_N
        alpha_max = (min_N * max_N - 1) / (min_N * max_N)

        logging.info(f"Alpha min/max {alpha_min}/{alpha_max}")

        if alpha == -1:
            # Determine a random alpha
            alpha = decimal.Decimal(random.uniform(alpha_min, alpha_max))
        else:
            alpha = min(max(alpha, alpha_min), alpha_max)

        k_alpha = max(1, math.ceil(((alpha - 1) * (min_N * max_N)) + min_N))

    logging.info(f"Alpha is {alpha} with k_alpha {k_alpha}")

    return alpha, k_alpha


def binary_search(q_list, k_alpha, affinity_matrix, matching_method,
                  apply_penalty=False,
                  penalty_step=0.1):
    left = 0
    right = len(q_list) - 1

    best_q = 0
    best_cost = 0
    best_avg_affinity = 0

    org_affinity = copy.deepcopy(affinity_matrix)
    org_q_list = copy.deepcopy(q_list)

    base_affinity = np.ones_like(affinity_matrix)
    best_matches = None
    penalty_coefficient = 1

    while left <= right:

        mid = (left + right) // 2
        mid_q = q_list[mid]

        if matching_method is not hungarian_cost_sensitive:
            cost_matrix = base_affinity * (affinity_matrix < mid_q)
            cost, matches = matching_method(cost_matrix)
        else:
            filtered_affinity = affinity_matrix * (affinity_matrix >= mid_q)
            cost, matches = hungarian_cost_sensitive(filtered_affinity)

        logging.info(f"Current mid is {mid_q} with cost {cost}")

        if cost <= k_alpha - 1:
            best_q = mid_q
            best_cost = cost
            best_matches = matches

            best_avg_affinity = sum(org_affinity[tuple(matches)]) / max(1, cost)

            left = mid + 1
        else:
            right = mid - 1

        if apply_penalty:
            penalty_coefficient += penalty_step
            affinity_matrix = org_affinity ** penalty_coefficient
            q_list = org_q_list ** penalty_coefficient

            nonzero_qlist = np.argwhere(q_list > 0)

            if len(nonzero_qlist) == 0:
                break

            left = max(nonzero_qlist[0][0], left)

    return best_q, best_cost, best_matches, affinity_matrix, best_avg_affinity


def quantile_registration(features_a,
                          features_b,
                          penalty_coefficient=1,
                          save_affinity=False,
                          alpha=-1,
                          method=hungarian_cost_sensitive):
    affinity_matrix = calculate_affinity(features_a, features_b, penalty_coefficient, save_affinity)

    alpha, k_alpha = find_k_alpha(affinity_matrix.shape, alpha)

    unique_q_list = np.unique(affinity_matrix)

    # Ensures the q_list has at least one element
    unique_q_list = unique_q_list[min(len(unique_q_list) - 1, k_alpha - 1):]

    best_q, cost, matches, affinity_matrix, max_affinity = binary_search(unique_q_list, k_alpha, affinity_matrix,
                                                                         method)

    # Weights are updated if penalty is applied
    weights = affinity_matrix[matches[0], matches[1]]

    # Sort the matches according to weights
    weight_indices = np.argsort(weights)[::-1]
    weights = weights[weight_indices]

    matches = matches[0][weight_indices], matches[1][weight_indices]

    return matches, weights, alpha, k_alpha, best_q, cost


if __name__ == "__main__":
    test_matrix = np.asarray([
        [36, 50, 68, 69],
        [72, 98, 9, 22],
        [79, 90, 6, 25],
        [2, 26, 70, 26],
        [62, 36, 38, 35],
        [27, 28, 28, 10],
        [55, 22, 21, 41],
    ])

    alpha = 0.8
    alpha, k_alpha = find_k_alpha(test_matrix.shape, alpha, reformulate=True)

    q_list = np.unique(test_matrix)[(k_alpha - 1):]

    best_q, cost, matches, affinity, avg_affinity = binary_search(q_list, k_alpha, test_matrix,
                                                                  matching_method=hungarian_cost_sensitive)

    for pair in zip(*matches):
        print(test_matrix[pair])

    print(f"Avg affinity: {avg_affinity} with cost: {cost}")
