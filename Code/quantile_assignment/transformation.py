import copy
import logging
import random

import numpy as np
import open3d
from numba import jit

from .visualizer import calculate_correspondence_lines


def decompose_transformation(transformation):
    rotation, translation = np.hsplit(np.vsplit(transformation, [3])[0], [3])

    return rotation, translation.T[0]


def calculate_transformation_matrix(rotation, translation):
    transformation_matrix = np.concatenate((rotation, np.asarray([translation]).T), axis=1)
    return np.concatenate((transformation_matrix, np.array([[0, 0, 0, 1]])), axis=0)


# Returns the inverse of the transformation which will register itself over itself
# Angles in radians
def apply_transformation(cloud,
                         angles,
                         translation):
    gt_rotation = cloud.get_rotation_matrix_from_xyz((angles[0], angles[1], angles[2]))
    gt_translation = np.asarray(translation)

    cloud.rotate(gt_rotation, center=(0, 0, 0)).translate(gt_translation)

    # We calculate rotation translation from A to B, therefore invert them
    inv_transformation = calculate_transformation_matrix(gt_rotation, gt_translation)
    inv_transformation = np.linalg.inv(inv_transformation)

    gt_rotation, gt_translation = decompose_transformation(inv_transformation)

    return gt_rotation, gt_translation


# Each trial tests 3 pairs, no pair is reevaluated in different trials
# Depending on the number of trials, some pairs may never be evaluated
def aggressive_tuple_test(cloud_a, cloud_b, matches, trials,
                          tau=0.9, seed=0,
                          visualize=False,
                          normal_alignment=False,  # Ignored
                          see_filtered=False) -> list:
    if visualize:
        cloud_a_copy = copy.deepcopy(cloud_a)
        cloud_b_copy = copy.deepcopy(cloud_b)

        cloud_a_copy.paint_uniform_color(np.asarray([0.0, 0.0, 1.0]))
        cloud_b_copy.paint_uniform_color(np.asarray([0.0, 0.0, 1.0]))

    points_a = np.asarray(cloud_a.points)
    points_b = np.asarray(cloud_b.points)

    matches_a = matches[0]
    matches_b = matches[1]

    filtered_indices = list(range(0, len(matches[0])))
    trials = min(trials * 3, len(matches[0])) // 3

    random.seed(seed)
    indices_list = random.sample(filtered_indices, trials * 3)

    false_color = np.array([1.0, 0.0, 0.0])
    true_color = np.array([0.0, 1.0, 0.0])
    assigned_color = false_color

    for i in range(trials):
        ind_1, ind_2, ind_3 = indices_list[i * 3: (i + 1) * 3]

        a_1, a_2, a_3 = matches_a[ind_1], matches_a[ind_2], matches_a[ind_3]
        b_1, b_2, b_3 = matches_b[ind_1], matches_b[ind_2], matches_b[ind_3]

        pa_1, pa_2, pa_3 = points_a[a_1], points_a[a_2], points_a[a_3]
        pb_1, pb_2, pb_3 = points_b[b_1], points_b[b_2], points_b[b_3]

        res_12 = np.linalg.norm(pa_1 - pa_2) \
                 / np.linalg.norm(pb_1 - pb_2)
        res_13 = np.linalg.norm(pa_1 - pa_3) \
                 / np.linalg.norm(pb_1 - pb_3)
        res_23 = np.linalg.norm(pa_2 - pa_3) \
                 / np.linalg.norm(pb_2 - pb_3)

        # Not a match
        if not tau < res_12 < (1 / tau) or not tau < res_13 < (1 / tau) or not tau < res_23 < (1 / tau):
            filtered_indices.remove(ind_1)
            filtered_indices.remove(ind_2)
            filtered_indices.remove(ind_3)

            assigned_color = false_color
            if not see_filtered:
                continue
        # Safe
        else:
            assigned_color = true_color

        if visualize:
            lines, spheres = calculate_correspondence_lines([[pa_1, pa_2, pa_3], [pb_1, pb_2, pb_3]],
                                                            line_color=assigned_color, sphere_radius=0.002)

            open3d.visualization.draw_geometries([cloud_a_copy, cloud_b_copy, lines, *spheres],
                                                 window_name="Our Tuple Test")

    return filtered_indices


# Runs for 100 * number of matches or maximum pair count is reached.
# Acts like downsampling with target, but a point can be evaluated again.
# In case the point passes the threshold, added to filtered matches. It can never be removed
# Normal alignment checks if the triangles are roughly aligned based on their normal directions: calculated
# according to order of the matches. Only recommended AFTER the initial registration
def fgr_tuple_test(cloud_a, cloud_b, matches, maximum_pair_count,
                   tau=0.9, seed=0,
                   visualize=False,
                   normal_alignment=False,
                   normal_alignment_threshold_degree=5,
                   see_filtered=False) -> list:
    if visualize:
        cloud_a_copy = copy.deepcopy(cloud_a)
        cloud_b_copy = copy.deepcopy(cloud_b)

        cloud_a_copy.paint_uniform_color(np.asarray([0.5, 0.0, 1.0]))
        cloud_b_copy.paint_uniform_color(np.asarray([0.0, 1.0, 1.0]))

    points_a = np.asarray(cloud_a.points)
    points_b = np.asarray(cloud_b.points)

    matches_a = matches[0]
    matches_b = matches[1]

    tuple_count = 0
    original_matches_count = len(matches[0])
    filtered_matches = set()

    random.seed(seed)
    trials = original_matches_count * 100

    false_color = np.array([1.0, 0.0, 0.0])
    true_color = np.array([0.0, 1.0, 0.0])

    for i in range(trials):

        # As a different range is given every time, same index can occur multiple times
        ind_1, ind_2, ind_3 = random.sample(range(0, original_matches_count), 3)

        a_1, a_2, a_3 = matches_a[ind_1], matches_a[ind_2], matches_a[ind_3]
        b_1, b_2, b_3 = matches_b[ind_1], matches_b[ind_2], matches_b[ind_3]

        pa_1, pa_2, pa_3 = points_a[a_1], points_a[a_2], points_a[a_3]
        pb_1, pb_2, pb_3 = points_b[b_1], points_b[b_2], points_b[b_3]

        res_12 = np.linalg.norm(pa_1 - pa_2) \
                 / np.linalg.norm(pb_1 - pb_2)
        res_13 = np.linalg.norm(pa_1 - pa_3) \
                 / np.linalg.norm(pb_1 - pb_3)
        res_23 = np.linalg.norm(pa_2 - pa_3) \
                 / np.linalg.norm(pb_2 - pb_3)

        normals_aligned = True
        normal_degree = 0

        if normal_alignment:
            edge_1a = pa_2 - pa_1
            edge_1a = edge_1a / np.linalg.norm(edge_1a)
            edge_2a = pa_3 - pa_1
            edge_2a = edge_2a / np.linalg.norm(edge_2a)
            normal_a = np.cross(edge_1a, edge_2a)
            normal_a = normal_a / np.linalg.norm(normal_a)

            edge_1b = pb_2 - pb_1
            edge_1b = edge_1b / np.linalg.norm(edge_1b)
            edge_2b = pb_3 - pb_1
            edge_2b = edge_2b / np.linalg.norm(edge_2b)
            normal_b = np.cross(edge_1b, edge_2b)
            normal_b = normal_b / np.linalg.norm(normal_b)

            normal_degree = np.degrees(np.arccos(np.dot(normal_a, normal_b)))
            normals_aligned = normal_degree < normal_alignment_threshold_degree

        # Match and normals are mostly aligned
        if tau < res_12 < (1 / tau) and tau < res_13 < (1 / tau) and tau < res_23 < (1 / tau) and normals_aligned:
            filtered_matches.add(ind_1)
            filtered_matches.add(ind_2)
            filtered_matches.add(ind_3)
            tuple_count += 1

            assigned_color = true_color
        else:
            assigned_color = false_color
            if not see_filtered:
                continue

        if visualize:
            lines, spheres = calculate_correspondence_lines([[pa_1, pa_2, pa_3], [pb_1, pb_2, pb_3]],
                                                            line_color=assigned_color, sphere_radius=0.002)

            open3d.visualization.draw_geometries([cloud_a_copy, cloud_b_copy, lines, *spheres],
                                                 window_name=f"FGR Tuple Test with Normal Degree: {normal_degree}")

        if tuple_count >= maximum_pair_count:
            break

    filtered_matches = list(filtered_matches)

    return filtered_matches


@jit(nopython=True)
def calculate_covariance(match_a,
                         match_b,
                         weights):
    covariance = np.zeros((3, 3))

    for c in range(match_a.shape[0]):
        cor_A = match_a[c].reshape(-1, 1)
        cor_B = match_b[c].reshape(-1, 1)

        covariance += weights[c] * cor_B.dot(cor_A.T)

    return covariance


def svd_transformation(source_cloud, target_cloud,
                       matches, weights=None):
    if weights is None:
        weights = np.ones(len(matches[0]))
    else:
        assert (len(weights) == len(matches[0]))

    # Calculate the transformation from correspondences
    match_A = np.asarray(source_cloud.points, dtype=np.float32)[matches[0]]
    match_B = np.asarray(target_cloud.points, dtype=np.float32)[matches[1]]

    # Normalize points according to their centers
    mean_A = np.average(match_A, axis=0)
    match_A = match_A - mean_A

    mean_B = np.average(match_B, axis=0)
    match_B = match_B - mean_B

    covariance = calculate_covariance(match_A,
                                      match_B,
                                      weights)

    logging.info(f"Covariance matrix: {covariance}")

    U, _, V_T = np.linalg.svd(covariance)

    rotation = U.dot(V_T)

    translation = mean_B - rotation.dot(mean_A)

    logging.info(f"Rotation: {rotation} and Translation: {translation}")

    return rotation, translation


def fgr_transformation(source_cloud, target_cloud,
                       matches, weights=None):
    # FGR optimization
    matches_arr = np.asarray(matches)
    matches_arr = matches_arr.T
    corr = open3d.utility.Vector2iVector(matches_arr)
    result = open3d.pipelines.registration.registration_fgr_based_on_correspondence(
        source_cloud, target_cloud, corr,
        open3d.pipelines.registration.FastGlobalRegistrationOption(tuple_test=False))

    return decompose_transformation(result.transformation)


def refine_transformation_icp(source_cloud, target_cloud,
                              initial_transformation,
                              distance_threshold):
    logging.info(f"Point-to-plane ICP registration is applied with distance threshold {distance_threshold}")
    result = open3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, distance_threshold, initial_transformation,
        open3d.pipelines.registration.TransformationEstimationPointToPlane())

    return decompose_transformation(result.transformation)


transformation_fnc = {
    "fgr": fgr_transformation,
    "svd": svd_transformation
}
