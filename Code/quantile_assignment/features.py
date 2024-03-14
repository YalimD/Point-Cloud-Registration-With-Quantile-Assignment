import logging

import numpy as np
import open3d
from open3d import geometry


def calculate_fpfh(point_cloud: geometry.PointCloud,
                   nearest_neighbour_radius: float,
                   nearest_neighbour_count: int):
    if nearest_neighbour_count is None and nearest_neighbour_radius is None:
        search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    elif nearest_neighbour_radius is None:
        search_param = open3d.geometry.KDTreeSearchParamKNN(nearest_neighbour_count)
    elif nearest_neighbour_count is None:
        search_param = open3d.geometry.KDTreeSearchParamRadius(nearest_neighbour_radius)
    else:
        search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=nearest_neighbour_radius,
                                                               max_nn=nearest_neighbour_count)

    pcd_fpfh = open3d.pipelines.registration.compute_fpfh_feature(point_cloud, search_param=search_param)

    return remove_nan_values(pcd_fpfh.data.T)


def pointLFSH(radius: float,
              bin_count: int,
              searchPoint_coordinate: np.ndarray,
              searchPoint_normal: np.ndarray,
              neighbor_points: np.ndarray,
              neighbor_normals: np.ndarray):
    # 3 histograms for our features
    # 10 for depth, 5 for density and 15 for angle as recommended by the paper
    depth_bin_count = int(bin_count / 3)
    depth_stride = 2 * radius / depth_bin_count
    depth_histogram = np.zeros(depth_bin_count)

    density_bin_count = int(bin_count / 6)
    density_stride = radius / density_bin_count
    density_histogram = np.zeros(density_bin_count)

    angle_bin_count = int(bin_count / 2)
    angle_stride = 180 / angle_bin_count
    angle_histogram = np.zeros(angle_bin_count)

    neighbour_count = neighbor_points.shape[0]
    for i in range(neighbour_count):

        neighbor_point = neighbor_points[i]
        neighbor_normal = neighbor_normals[i]

        # region Depth

        # Direct formula from the paper
        temp_depth = radius - np.dot(searchPoint_normal, neighbor_point - searchPoint_coordinate)

        # Compute histograms
        if temp_depth >= 2 * radius:
            depth_bin_id = depth_bin_count
        elif temp_depth <= 0:
            depth_bin_id = 1
        else:
            depth_bin_id = temp_depth / depth_stride + 1

        # endregion

        # region Radius

        s2n = neighbor_point - searchPoint_coordinate
        c = np.dot(s2n, s2n)
        b = np.dot(s2n, searchPoint_normal)

        temp_radius = np.sqrt(abs(c - b * b))

        if temp_radius >= radius:
            density_bin_id = density_bin_count
        else:
            density_bin_id = temp_radius / density_stride + 1

        # endregion

        # region Angle

        temp_angle = np.dot(searchPoint_normal, neighbor_normal)
        temp_angle = max(min(temp_angle, 1), -1)
        temp_angle = np.arccos(temp_angle) / np.pi * 180

        if temp_angle >= 180:
            angle_bin_id = angle_bin_count
        else:
            angle_bin_id = temp_angle / angle_stride + 1

        # endregion

        depth_bin_id = int(depth_bin_id)
        density_bin_id = int(density_bin_id)
        angle_bin_id = int(angle_bin_id)

        # Normalize according to point count

        depth_histogram[depth_bin_id - 1] += 1 / (neighbor_points.shape[0])
        density_histogram[density_bin_id - 1] += 1 / (neighbor_points.shape[0])
        angle_histogram[angle_bin_id - 1] += 1 / (neighbor_points.shape[0])

    histogram = np.append(depth_histogram, density_histogram)
    histogram = np.append(histogram, angle_histogram)

    return histogram


def calculate_lfsh(point_cloud: geometry.PointCloud,
                   nearest_neighbour_radius: float,
                   nearest_neighbour_count: int,
                   bin_count: int = 30):
    """
    :param point_cloud: The point cloud
    :param nearest_neighbour_radius: Neighbor radius
    :param nearest_neighbour_count:  Neighbor count ignored
    :param bin_count: Number of bins, 30 is recommended
    :return: histogram of features
    """

    # We expect normals to be calculated
    if not point_cloud.has_points() or not point_cloud.has_normals():
        logging.error("LFSH expects normals")
        return None

    points = np.array(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    histograms = np.zeros((points.shape[0], bin_count))

    kd_tree = open3d.geometry.KDTreeFlann(point_cloud)

    for point_index in range(points.shape[0]):

        query_point = points[point_index]
        query_normal = normals[point_index]

        # No need to give number of neighbours
        [neighbor_count, neighbor_indices, _] = kd_tree.search_radius_vector_3d(query_point,
                                                                                nearest_neighbour_radius)
        neighbor_indices.remove(point_index)
        neighbor_count = neighbor_count - 1

        assert (neighbor_count == len(neighbor_indices))

        sphere_neighbors = np.zeros((len(neighbor_indices), 3))
        sphere_normals = np.zeros((len(neighbor_indices), 3))

        # Expect at least 2 neighbours, excluding myself ofc
        if neighbor_count > 2:
            for j in range(len(neighbor_indices)):
                sphere_neighbors[j] = points[neighbor_indices[j]]
                sphere_normals[j] = normals[neighbor_indices[j]]

            # LRA = np.array((normals[indices[i]][0], normals[indices[i]][1], normals[indices[i]][2]))

            histograms[point_index] = pointLFSH(nearest_neighbour_radius, bin_count, query_point, query_normal,
                                                sphere_neighbors, sphere_normals)
        else:
            histograms[point_index] = np.zeros(bin_count)  # default feature values

    return remove_nan_values(histograms)


def remove_nan_values(my_array):
    return my_array[~np.isnan(my_array).any(axis=1)]
