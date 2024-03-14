import numpy
import copy
import open3d
import os.path
import datetime
import numpy as np

def convert_to_quaternion(rotation):

    quaternion = numpy.zeros(4)

    quaternion[0] = 0.5 * numpy.sqrt(1 + rotation[0][0] + rotation[1][1] + rotation[2][2])
    quaternion[1] = -(rotation[2][1] - rotation[1][2]) / (4 * quaternion[0])
    quaternion[2] = -(rotation[0][2] - rotation[2][0]) / (4 * quaternion[0])
    quaternion[3] = -(rotation[1][0] - rotation[0][1]) / (4 * quaternion[0])

    return quaternion


def calculate_diameter(pcd):
    diameter = numpy.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    return diameter


def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = numpy.asarray(noisy_pcd.points)
    points += numpy.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = open3d.utility.Vector3dVector(points)
    return noisy_pcd


def add_noise(dataset_root, dataset_name, partial_no):
    level = 0
    path = os.path.join(f"{dataset_root}{dataset_name}", f"{dataset_name}_noise_level_{level}", f"{dataset_name}_noise_level_{level}_partial_{partial_no}.ply")
    pcd = open3d.io.read_point_cloud(path)
    pcd_diameter = calculate_diameter(pcd)

    noise_sigma = [0.0025, 0.005]

    for sigma in noise_sigma:
        level += 1
        noisy_pcd = apply_noise(pcd, 0, sigma * pcd_diameter)
        path = os.path.join(f"{dataset_root}{dataset_name}", f"{dataset_name}_noise_level_{level}",
                            f"{dataset_name}_noise_level_{level}_partial_{partial_no}.ply")
        open3d.io.write_point_cloud(path, noisy_pcd)


def calculate_overlap_ratio(source_cloud: open3d.geometry.PointCloud,
                            target_cloud: open3d.geometry.PointCloud,
                            distance_threshold: float = 0) -> float:

    source_size = len(numpy.asarray(source_cloud.points))
    target_size = len(numpy.asarray(target_cloud.points))

    small_cloud, large_cloud = (source_cloud, target_cloud) if source_size < target_size else (target_cloud,
                                                                                               source_cloud)
    kd_tree = open3d.geometry.KDTreeFlann(large_cloud)

    small_points = numpy.asarray(small_cloud.points)
    large_points = numpy.asarray(large_cloud.points)

    count = 0
    for point in small_points:
        [_, neighbor_indices, _] = kd_tree.search_knn_vector_3d(point, 1)

        # Expect exactly one neighbour
        neighbour = large_points[neighbor_indices[0]]

        if numpy.linalg.norm(point - neighbour) <= distance_threshold:
            count += 1

    return count / len(small_points)

def generate_partials(pcd): #bunny için

    partial_1 = pcd.crop(
        open3d.geometry.OrientedBoundingBox(center=np.array([0.0, 0.15, 0.05]), R=np.eye(3), extent=np.array([0.15] * 3)))
    partial_2 = pcd.crop(
        open3d.geometry.OrientedBoundingBox(center=np.array([0.0, -0.0, 0.0]), R=np.eye(3), extent=np.array([0.25] * 3)))
    partial_3 = pcd.crop(
        open3d.geometry.OrientedBoundingBox(center=np.array([0.0, 0.3, 0.0]), R=np.eye(3), extent=np.array([0.4] * 3)))
    partial_4 = pcd.crop(
        open3d.geometry.OrientedBoundingBox(center=np.array([0.0, 0.15, 0.0]), R=np.eye(3), extent=np.array([0.1, 0.2, 0.2])))
    partial_5 = pcd.crop(
        open3d.geometry.OrientedBoundingBox(center=np.array([-0.1, -0.0, 0.0]), R=np.eye(3), extent=np.array([0.2, 0.3, 0.3])))

    return partial_1, partial_2, partial_3, partial_4, partial_5


def create_timed_folder(path):
    now = datetime.datetime.now()
    path = os.path.join(path, now.strftime("%Y-%m-%d_%H-%M-%S"))

    if not os.path.exists(path):
        os.makedirs(path)

    return path

#if __name__ == "__main__":
#     dataset_root = "../../Data/Dummy/"
#     add_noise(dataset_root, "horse", 1)



