import os

import omegaconf.errors
from omegaconf import DictConfig
from tqdm import tqdm

from Code.quantile_assignment import *
from .utils import *


# Papers:
# The data used here is the 8 scenes included in 3DMatch
# The original scenes are from :
# - 7-scenes
# - sun3D
# https://3dmatch.cs.princeton.edu/
# The testing protocol is from:
# http://redwood-data.org/indoor/registration.html


def dataset_format(partition_S, partition_T, num_of_partitions,
                   transformation):
    report = f"{partition_T}\t{partition_S}\t{num_of_partitions}\n"
    for row in range(4):
        report += f"{transformation[row][0]:.8E}\t{transformation[row][1]:.8E}\t" \
                  f"{transformation[row][2]:.8E}\t{transformation[row][3]:.8E}\n"
    return report

# region gt_processing


def process_gt_log(gt_file):
    match_table = None
    accepted_num_of_matches = 0
    total_num_of_matches = 0

    with open(gt_file, "r") as log_reader:
        while log_reader.readable():
            line = log_reader.readline()
            if len(line) == 0:
                break
            target, source, number_of_fragments = map(lambda term: int(term), line.split())

            if match_table is None:
                match_table = [None] * math.ceil(number_of_fragments * (number_of_fragments + 1) / 2)

            transformation = np.zeros((4, 4))
            for r in range(4):
                row = list(map(lambda term: float(term), log_reader.readline().split()))
                transformation[r] = np.asarray(row)

            match_table[int(source * (source + 1) / 2) + target] = transformation

            total_num_of_matches += 1
            if source > target + 1:
                accepted_num_of_matches += 1

    return match_table, accepted_num_of_matches, total_num_of_matches


def process_gt_info(gt_info):
    info_table = None
    accepted_num_of_matches = 0
    total_num_of_matches = 0

    with open(gt_info, "r") as info_reader:
        while info_reader.readable():
            line = info_reader.readline()
            if len(line) == 0:
                break
            target, source, number_of_fragments = map(lambda term: int(term), line.split())

            if info_table is None:
                info_table = [None] * math.ceil(number_of_fragments * (number_of_fragments + 1) / 2)

            info_data = np.zeros((6, 6))
            for r in range(6):
                row = list(map(lambda term: float(term), info_reader.readline().split()))
                info_data[r] = np.asarray(row)

            info_table[int(source * (source + 1) / 2) + target] = info_data

            total_num_of_matches += 1
            if source > target + 1:
                accepted_num_of_matches += 1

    return info_table, accepted_num_of_matches, total_num_of_matches


def calculate_rmse(transformation, gt_trans, info_mat):
    joint = np.linalg.inv(gt_trans) @ transformation

    translation = joint[0:3, 3]
    rotation = joint[:3, :3]

    q_rot = convert_to_quaternion(rotation)
    rho = np.asarray([*translation, *(-q_rot[1:])])

    return (rho.T @ info_mat @ rho) / info_mat[0][0]


def evaluate_fragment_registration(results_file_name, gt_folder, rmse_distance_threshold=0.2):
    gt_file = os.path.join(gt_folder, "gt.log")
    gt_info = os.path.join(gt_folder, "gt.info")

    log_table, accepted_number_of_matches, _ = process_gt_log(gt_file)
    info_table, *_ = process_gt_info(gt_info)

    good_matches = 0
    total_matches = 0

    # Find the number of correct registrations from gt. The indices must be non-consecutive, ignore any such
    with open(results_file_name, "r") as result_reader:
        while result_reader.readable():
            line = result_reader.readline()
            if len(line) == 0:
                break

            target, source, number_of_fragments = map(lambda term: int(term), line.split())

            # Only consider the non-consecutive matches
            if source > target + 1:
                total_matches += 1

                # If it is also a match in gt
                current_index = int(source * (source + 1) / 2) + target
                gt_trans = log_table[current_index]
                info_mat = info_table[current_index]

                if gt_trans is not None and info_mat is not None:

                    transformation = np.zeros((4, 4))
                    for r in range(4):
                        row = list(map(lambda term: float(term), result_reader.readline().split()))
                        transformation[r] = np.asarray(row)

                    rmse_p = calculate_rmse(transformation, gt_trans, info_mat)
                    if rmse_p <= np.power(rmse_distance_threshold, 2):
                        good_matches += 1
                else:  # Skip 4 lines
                    for r in range(4):
                        result_reader.readline()
            else:  # Skip 4 lines
                for r in range(4):
                    result_reader.readline()

    recall = good_matches / accepted_number_of_matches
    precision = good_matches / max(total_matches, 1)

    return recall, precision


def sanity_test(path_to_data):

    # Sanity test
    path_to_3dmatch = os.path.join(path_to_data, "7-scenes-redkitchen-evaluation", "3dmatch.log")
    path_to_gt = os.path.join(path_to_data, "7-scenes-redkitchen-evaluation")

    recall, precision = evaluate_fragment_registration(path_to_3dmatch, path_to_gt)
    logging.info(f"Recall: {recall} and Precision: {precision} for 3Dmatch on redkitchen")

    assert (np.isclose(recall, 0.85300) and np.isclose(precision, 0.72128))


#endregion

def run_fragment(cfg: DictConfig,
                 cloud_count, data_extension, fragment_dir,
                 result_folder, results_file_name,
                 method) -> float:

    if not cfg.method.cloud_from_path:
        try:
            downsampler = downsampler_fcn[cfg.method.downsampler]
        except omegaconf.errors.ConfigAttributeError:
            downsampler = None
    else:
        downsampler = None

    base_file_name = "cloud_bin_"

    visualization_path = None
    duration = 0

    with open(results_file_name, "w") as result_writer:
        for target_no in tqdm(range(cloud_count)):
            target_cloud_path = os.path.join(fragment_dir, f"{base_file_name}{target_no}{data_extension}")
            original_target = read_cloud(target_cloud_path)

            target_cloud = copy.deepcopy(original_target)
            if downsampler is not None:
                target_cloud = downsampler(target_cloud, *cfg.method.downsampler_args)

            for source_no in range(target_no + 1, cloud_count):
                source_cloud_path = os.path.join(fragment_dir, f"{base_file_name}{source_no}{data_extension}")
                original_source = read_cloud(source_cloud_path)

                source_cloud = copy.deepcopy(original_source)
                if downsampler is not None:
                    source_cloud = downsampler(source_cloud, *cfg.method.downsampler_args)

                visualization_path = os.path.join(result_folder, f"visuals_{source_no}_{target_no}")

                if cfg.method.cloud_from_path:
                    registration_duration, resulting_transformation, result_txt = method.run(source_cloud_path,
                                                                                             target_cloud_path,
                                                                                             visualization_path)
                    logging.info(result_txt)
                else:
                    # TODO: For quantile, metrics are ignored. Check if true
                    registration_duration, resulting_transformation, _ = method.run(source_cloud,
                                                                                 target_cloud,
                                                                                 visualization_path)
                duration += registration_duration

                # Determine overlap to deem successful registration
                # According to paper, overlap below %30 shouldn't be accepted
                # NOTE: We calculate overlap according to original clouds
                original_source = original_source.transform(resulting_transformation)

                # This is from paper
                overlap_ratio = calculate_overlap_ratio(original_source, original_target,
                                                        distance_threshold=cfg.dataset.registration.neighbour_distance)

                if overlap_ratio >= cfg.dataset.registration.overlap_ratio_threshold:
                    report = dataset_format(source_no, target_no, cloud_count,
                                            resulting_transformation)
                    result_writer.write(report)

                result_writer.flush()

    return duration


def run_3dmatch(cfg: DictConfig, method) -> None:

    path_to_data = cfg.dataset.data_path
    sanity_test(path_to_data)

    path_to_results = cfg.dataset.result_path

    cwd = os.getcwd()
    os.chdir(path_to_data)
    fragment_list = [path for path in os.listdir() if not os.path.isfile(path) and not path.endswith("evaluation")]
    gt_list = [path for path in os.listdir() if not os.path.isfile(path) and path.endswith("evaluation")]
    os.chdir(cwd)

    recall_list = []
    precision_list = []
    duration = 0

    for fragment, evaluation in zip(fragment_list, gt_list):

        fragment_dir = os.path.join(path_to_data, fragment)
        evaluation_dir = os.path.join(path_to_data, evaluation)

        os.chdir(fragment_dir)
        cloud_list = os.listdir()
        cloud_count = len(os.listdir())
        data_extension = os.path.splitext(cloud_list[0])[1]
        os.chdir(cwd)

        result_folder = os.path.join(os.path.join(path_to_results, fragment))
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        results_file_name = os.path.join(result_folder, "result.log")

        # If the result file exists, regardless it is complete or not, evaluate it and skip processing the data
        if not os.path.exists(results_file_name):
            duration += run_fragment(cfg, cloud_count, data_extension, fragment_dir,
                                     result_folder, results_file_name, method)

        recall, precision = evaluate_fragment_registration(results_file_name, evaluation_dir)

        recall_list.append(recall)
        precision_list.append(precision)

    recall_avg = np.average(recall_list)
    precision_avg = np.average(precision_list)

    with open(os.path.join(path_to_results, "artifacts.txt"), "w") as artifact_writer:
        artifact_writer.write(f"{duration}(s)\n")
        artifact_writer.write(f"Recalls: {recall_list}.\t Avg: {recall_avg}\n")
        artifact_writer.write(f"Precisions: {precision_list}.\t Avg: {precision_avg}")

    logging.info(f"Recall: {recall_avg} and Precision: {precision_avg}")