import numpy as np
import open3d
import scipy.io


def process_ply(file_path):
    return open3d.io.read_point_cloud(file_path)


# Logic from: https://github.com/qinzheng93/GeoTransformer/blob/main/data/Kitti/downsample_pcd.py
def process_kitti(file_path):
    # Last axis is intensity which we ignore
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]

    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points)

    return cloud


def process_off(file_path):
    mesh = open3d.io.read_triangle_mesh(file_path)
    return open3d.geometry.PointCloud(points=mesh.vertices)


def process_mat(file_path):
    pc = scipy.io.loadmat(file_path)

    variable_name = list(filter(lambda key: key.find("__") < 0, pc.keys()))[-1]

    points = open3d.utility.Vector3dVector(pc[variable_name].T)

    return open3d.geometry.PointCloud(points=points)


extension_fcn = {
    ".ply": process_ply,
    ".kitti": process_kitti,
    ".off": process_off,
    ".mat": process_mat
}
