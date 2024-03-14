import copy
import logging

import numpy as np
from numba import jit


def calculate_TE(trs, gt_trs):
    if gt_trs is None or gt_trs == "None":
        logging.warning("Ground truth translation is not provided")
        return None
    return np.linalg.norm(trs - gt_trs)


def calculate_RE(rot, gt_rot):
    if gt_rot is None or gt_rot == "None":
        logging.warning("Ground truth rotation is not provided")
        return None
    return np.rad2deg(np.arccos((np.trace(rot.T @ gt_rot) - 1) / 2))


@jit(nopython=True)
def calculate_rooted_distance(s_points,
                              t_points):
    total_distance = 0
    for p in range(len(s_points)):
        total_distance += np.power(np.linalg.norm(s_points[p] - t_points[p]), 2)

    return total_distance


def calculate_rmseP(source_cloud, rot, trs,
                    target_cloud, gt_cor):
    if gt_cor is None or gt_cor == "None":
        logging.warning("Ground truth correspondences are not provided")
        return None

    source_cloud_transformed = copy.deepcopy(source_cloud)
    source_cloud_transformed = source_cloud_transformed.rotate(rot, center=(0, 0, 0)).translate(trs)

    source_cloud_points = np.asarray(source_cloud_transformed.points)
    target_cloud_points = np.asarray(target_cloud.points)

    source_indices, target_indices = gt_cor

    total_distance = calculate_rooted_distance(source_cloud_points[source_indices],
                                               target_cloud_points[target_indices])

    return np.sqrt(total_distance / len(source_indices))


def evaluate_registration(rot, trs,
                          gt_rot, gt_trs,
                          source_cloud, target_cloud,
                          gt_cor,
                          projection_error_threshold_meters: float = 0.2,
                          rotation_threshold_degrees: float = 5.0,
                          translation_threshold_meters: float = 2.0):
    metrics = {
        "rmseP": calculate_rmseP(source_cloud, rot, trs,
                                 target_cloud, gt_cor),
        "RE": calculate_RE(rot, gt_rot),
        "TE": calculate_TE(trs, gt_trs)
    }

    z = list(zip(list(metrics.values()), [projection_error_threshold_meters,
                                          rotation_threshold_degrees,
                                          translation_threshold_meters]))
    success = all(list(map(lambda t: True if t[0] is None else t[0] <= t[1], z)))

    return metrics, success
