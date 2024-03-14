import logging
import os.path
from typing import Optional

from .fileReader import *


def read_cloud(file_path: str) -> Optional[open3d.geometry.PointCloud]:
    """Reads the point cloud from file based on extension and returns it as open3d structure"""

    base_path = os.path.basename(file_path)

    logging.info(f"Loading the point cloud from {base_path}")

    extension = os.path.splitext(file_path)[1]
    point_cloud = extension_fcn[extension](file_path)
    point_count = np.asarray(point_cloud.points).shape[0]

    if point_count == 0:
        logging.error(f"Point cloud {base_path} is empty!")
        return None

    logging.info(f"Number of points for {base_path}: {point_count}")
    logging.info(f"Bounds min/max: {point_cloud.get_min_bound()}/{point_cloud.get_max_bound()}")
    logging.info(f"Diameter: {np.linalg.norm(point_cloud.get_max_bound() - point_cloud.get_min_bound())}")

    return point_cloud


def calculate_normals(point_cloud: open3d.geometry.PointCloud,
                      nearest_neighbour_radius: float,
                      nearest_neighbour_count: int) -> Optional[open3d.geometry.PointCloud]:
    if nearest_neighbour_count is None and nearest_neighbour_radius is None:
        search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    elif nearest_neighbour_radius is None:
        search_param = open3d.geometry.KDTreeSearchParamKNN(nearest_neighbour_count)
    elif nearest_neighbour_count is None:
        search_param = open3d.geometry.KDTreeSearchParamRadius(nearest_neighbour_radius)
    else:
        search_param = open3d.geometry.KDTreeSearchParamHybrid(radius=nearest_neighbour_radius,
                                                               max_nn=nearest_neighbour_count)

    point_cloud.estimate_normals(search_param)
    nearest_neighbour_count = nearest_neighbour_count or 30
    point_cloud.orient_normals_consistent_tangent_plane(nearest_neighbour_count)

    normal_count = np.asarray(point_cloud.normals).shape[0]

    if normal_count == 0:
        logging.error(f"No normals could be calculated!")

    return point_cloud
