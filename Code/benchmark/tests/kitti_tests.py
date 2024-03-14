import glob
import os
import os.path as osp

import open3d as o3d
from omegaconf import DictConfig
from tqdm.auto import tqdm

from Code.quantile_assignment import *
from .utils import *


# For testing, we will only register pairs that are at least 10 meters apart from each other.
# This code is heavily based on: https://github.com/qinzheng93/GeoTransformer/blob/main/data/Kitti/downsample_pcd.py
# and https://github.com/chrischoy/FCGF/blob/master/lib/data_loaders.py

# region FCGF dataloader lib

def _odometry_to_positions(odometry):
    T_w_cam0 = odometry.reshape(3, 4)
    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
    return T_w_cam0


def _rot3d(self, axis, angle):
    ei = np.ones(3, dtype='bool')
    ei[axis] = 0
    i = np.nonzero(ei)[0]
    m = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)
    m[i[0], i[0]] = c
    m[i[0], i[1]] = -s
    m[i[1], i[0]] = s
    m[i[1], i[1]] = c
    return m


def _pos_transform(self, pos):
    x, y, z, rx, ry, rz, _ = pos[0]
    RT = np.eye(4)
    RT[:3, :3] = np.dot(np.dot(self.rot3d(0, rx), self.rot3d(1, ry)), self.rot3d(2, rz))
    RT[:3, 3] = [x, y, z]
    return RT


def _get_position_transform(self, pos0, pos1, invert=False):
    T0 = self.pos_transform(pos0)
    T1 = self.pos_transform(pos1)
    return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
        np.linalg.inv(T1), T0).T)


# https://github.com/chrischoy/FCGF/blob/0612340ead256adb5449da8088f506e947e44b4c/scripts/train_fcgf_kitti.sh#L15
def get_matching_indices(source, target, search_voxel_size=1.5 * 0.3, K=None):
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_inds = []
    for i, point in enumerate(source.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

# endregion


def _preprocess_cloud(cloud_path, voxel_downsample_size):
    frame = cloud_path.split('/')[-1][:-4]
    new_file_name = osp.join(frame + '_downsampled.ply')

    points = np.fromfile(cloud_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    original_cloud = copy.deepcopy(pcd)

    if not os.path.exists(new_file_name):
        pcd = pcd.voxel_down_sample(voxel_downsample_size)
        o3d.io.write_point_cloud(new_file_name, pcd)
    return original_cloud


def _preprocess_dataset(cloud_data_path, position_data_path,
                        test_sequences, cfg, test_pairs_path, corrected_transformation_path,
                        icp_overlap_count=1000):
    if os.path.exists(test_pairs_path) and os.path.exists(corrected_transformation_path):
        logging.warning(f"Skipping preprocess since {test_pairs_path} and {corrected_transformation_path} exists")

        pairs = np.genfromtxt(test_pairs_path).reshape(-1, 3)

        transformations = np.genfromtxt(corrected_transformation_path).reshape(-1, 16)
        transformations = [transformation.reshape(4, 4) for transformation in transformations]

        return pairs, transformations

    problematic_pairs = [(8, 15, 58)]

    R = np.array([
        7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
        -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
    ]).reshape(3, 3)
    T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    velo2cam = np.hstack([R, T])
    velo2cam = np.vstack((velo2cam, [0, 0, 0, 1]))

    pairs = []
    transformations = []

    for sequence in tqdm(test_sequences):

        # Read the transformations
        seq_id = '{:02d}'.format(int(sequence))
        fnames = glob.glob(osp.join(cloud_data_path, seq_id, 'velodyne', '*.bin'))
        assert len(fnames) > 0, f"Make sure that the path {cloud_data_path} has data {seq_id}"
        inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

        all_odo = np.genfromtxt(os.path.join(position_data_path, "{}.txt".format(seq_id)))
        all_pos = np.array([_odometry_to_positions(odo) for odo in all_odo])

        Ts = all_pos[:, :3, 3]
        pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
        pdist = np.sqrt(pdist.sum(-1))
        valid_pairs = pdist > cfg.dataset.distance_between_pairs
        curr_time = inames[0]

        while curr_time in inames:
            # Find the min index
            next_time = np.where(valid_pairs[curr_time][curr_time:curr_time + cfg.dataset.time_between_pairs])[0]
            if len(next_time) == 0:
                curr_time += 1
            else:
                # Follow https://github.com/yewzijian/3DFeatNet/blob/master/scripts_data_processing/kitti/process_kitti_data.m#L44
                next_time = next_time[0] + curr_time - 1

            if next_time in inames:
                pair = (int(sequence), curr_time, next_time)
                if pair not in problematic_pairs:

                    # Both clouds are original
                    current_cloud = _preprocess_cloud(fnames[curr_time],
                                                      cfg.dataset.voxel_downsample_size)
                    next_cloud = _preprocess_cloud(fnames[next_time],
                                                   cfg.dataset.voxel_downsample_size)

                    # Downsample again using the values from FCGF
                    current_cloud = current_cloud.voxel_down_sample(0.05)
                    next_cloud = next_cloud.voxel_down_sample(0.05)

                    # This is in column major
                    transformation = (np.linalg.inv(velo2cam) @ np.linalg.inv(all_pos[next_time]) @ all_pos[
                        curr_time] @ velo2cam)

                    current_corrected = current_cloud.transform(transformation)

                    reg = o3d.pipelines.registration.registration_icp(
                        current_corrected, next_cloud, 0.2, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

                    transformation = reg.transformation @ transformation
                    current_corrected = current_cloud.transform(reg.transformation)

                    matches = get_matching_indices(current_corrected, next_cloud)
                    if len(matches) < icp_overlap_count:
                        logging.error(f"Invalid pair: {pair}")
                    else:
                        pairs.append(np.array(pair))
                        transformations.append(transformation)
                        logging.info(f"Added {pair}. Current count is {len(pairs)}")

                curr_time = next_time + 1

    # From paper
    assert (len(pairs) == len(transformations) == 555)

    np.savetxt(test_pairs_path, np.array(pairs), fmt='%d')
    np.savetxt(corrected_transformation_path, np.array(transformations).reshape(-1, 16))

    return pairs, transformations


def run_sequences(cfg: DictConfig, method,
                  pairs, transformations):
    if pairs is None or transformations is None:
        raise ValueError("Pairs and transformation cannot be empty")

    result_root = cfg.dataset.result_path

    total_duration = 0
    total_success = 0
    rre_list = []
    rte_list = []

    for pair, trans in tqdm(zip(pairs, transformations), total=len(pairs)):
        # Read the pair
        sequence_id, source_id, target_id = [int(element) for element in pair]
        sequence_path = os.path.join(cfg.dataset.data_path, "sequences", "{:02d}".format(sequence_id), "velodyne")

        source_path = os.path.join(sequence_path, "{:06d}_downsampled.ply".format(source_id))
        target_path = os.path.join(sequence_path, "{:06d}_downsampled.ply".format(target_id))

        logging.info(f"{'-' * 20} Source: {source_id} - Target:{target_id} {'-' * 20}")

        source_cloud = open3d.io.read_point_cloud(source_path)
        target_cloud = open3d.io.read_point_cloud(target_path)

        visualization_path = os.path.join(result_root,
                                          f"seq{int(sequence_id)}_pair{int(source_id)}-{int(target_id)}")

        downsampler = downsampler_fcn[cfg.method.downsampler]
        source_cloud = downsampler(source_cloud, *cfg.method.downsampler_args)
        target_cloud = downsampler(target_cloud, *cfg.method.downsampler_args)

        gt_rotation, gt_translation = trans[:3, :3], trans[:3, 3]

        registration_duration, resulting_transformation, _ = method.run(source_cloud,
                                                                        target_cloud,
                                                                        visualization_path)

        rotation, translation = decompose_transformation(resulting_transformation)
        errors, success = evaluate_registration(rotation, translation,
                                                gt_rotation, gt_translation,
                                                None, None, None,
                                                rotation_threshold_degrees=cfg.dataset.rre_threshold_degrees,
                                                translation_threshold_meters=cfg.dataset.rte_threshold_meters)

        if errors["RE"] < cfg.dataset.rre_threshold_degrees:
            rre_list.append(errors["RE"])

        if errors["TE"] < cfg.dataset.rte_threshold_meters:
            rte_list.append(errors["TE"])

        logging.info(f"Success: {success}. RRE: {errors['RE']}. RTE: {errors['TE']}")
        logging.info(f"Transformation: {resulting_transformation}")

        total_duration += registration_duration
        total_success += success

    logging.info(f"Duration: {total_duration}")

    logging.info(f"RRE Mean:{np.mean(rre_list)} Std:{np.std(rre_list)}")
    logging.info(f"RTE Mean:{np.mean(rte_list)} Std:{np.std(rte_list)}")

    logging.info(f"Success Rate: {100 * total_success / len(pairs)}")


def run_kitti(cfg: DictConfig, method):

    test_sequences_path = os.path.join(cfg.dataset.data_path,
                                       cfg.dataset.test_sequences)

    with open(test_sequences_path, "r") as test_reader:
        test_sequences = test_reader.read().split()

    cloud_data_path = os.path.join(cfg.dataset.data_path,
                                   "sequences")

    position_data_path = os.path.join(cfg.dataset.data_path, "poses")

    test_pairs_path = os.path.join(cfg.dataset.data_path, cfg.dataset.test_pairs)
    corrected_transformation_path = os.path.join(cfg.dataset.data_path, cfg.dataset.corrected_transformation)

    pairs, transformations = _preprocess_dataset(cloud_data_path, position_data_path,
                                                 test_sequences, cfg, test_pairs_path, corrected_transformation_path,
                                                 icp_overlap_count=1000)

    run_sequences(cfg, method, pairs, transformations)
