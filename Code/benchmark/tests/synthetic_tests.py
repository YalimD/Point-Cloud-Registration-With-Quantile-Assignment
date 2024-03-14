import os
import os.path

from omegaconf import DictConfig, OmegaConf

import Code.quantile_assignment as qa
from .utils import *


def custom_format(partition_S, partition_T, num_of_partitions,
                  transformation,
                  downsample_method, downsample_parameter,
                  alpha, errors, success, duration,
                  additional_report):

    report = "*" * 50
    report += "REPORT START"
    report += "*" * 50
    report += "\n"

    report += f"{partition_S} {partition_T} {num_of_partitions}\n"
    report += f"{transformation}\n"
    report += f"Downsample method/parameter: {downsample_method}/{downsample_parameter}\n"
    report += f"Alpha: {alpha}\n"
    report += f"Errors: {errors}\n"
    report += f"Success: {success}. Duration(s): {duration}\n"
    report += "Additional report:\n"
    report += additional_report

    report += "*" * 50
    report += "REPORT END"
    report += "*" * 50
    report += "\n"

    return report


def run_partial_matches(cfg: DictConfig,
                        method):
    dataset_root = os.path.abspath(cfg.dataset.data_path)
    result_root = os.path.abspath(cfg.dataset.result_path)

    noise_levels = range(3)

    downsampler_args = cfg.method.downsampler_args

    for object in os.listdir(dataset_root):

        object_path = os.path.join(dataset_root, object)

        for noise_level in noise_levels:
            for down_param in downsampler_args:

                cfg.method.downsampler_args = down_param

                base_file_name = f"{object}_noise_level_{noise_level}"
                data_path = os.path.join(object_path, base_file_name)

                result_folder = os.path.join(result_root, object,
                                             f"noise_level_{noise_level}", f"down_param_{down_param}")

                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)

                results_file_name = os.path.join(result_folder, "report.txt")

                partial_numbers = [os.path.splitext(no)[0].split("_")[-1] for no in os.listdir(data_path) if
                                   os.path.isfile(os.path.join(data_path, no))]

                total_success = 0
                total_pairs = 0

                temp_src_path = os.path.join(result_root, "temp_src.ply")

                with open(results_file_name, "w") as report_writer:
                    for partial_no_S in partial_numbers:
                        source_cloud_path = os.path.join(data_path, f"{base_file_name}_partial_{partial_no_S}.ply")
                        source_cloud = qa.read_cloud(source_cloud_path)
                        original_source = copy.deepcopy(source_cloud)

                        rotation_angles = cfg.dataset.transformation.rotation_degrees
                        applied_transformation = cfg.dataset.transformation.translation

                        applied_rotation = list(map(lambda r: np.radians(r), rotation_angles))

                        gt_rotation, gt_translation = qa.apply_transformation(source_cloud, applied_rotation,
                                                                              applied_transformation)

                        # If the pc needs to be as file, export this source first for speed
                        if cfg.method.cloud_from_path:
                            # ADHOC: The output should be in ascii format and use single precision
                            open3d.io.write_point_cloud(temp_src_path, source_cloud, write_ascii=True)
                            with open(temp_src_path, "r") as float_corrector:
                                t = float_corrector.read()
                            t = t.replace("double", "float")
                            with open(temp_src_path, "w") as float_corrector:
                                float_corrector.write(t)

                        for partial_no_T in partial_numbers:
                            if partial_no_S < partial_no_T:
                                target_cloud_path = os.path.join(data_path,
                                                                 f"{base_file_name}_partial_{partial_no_T}.ply")
                                original_target = qa.read_cloud(target_cloud_path)

                                cfg.method.alpha = cfg.dataset.overlap[object]

                                visualization_path = os.path.join(result_folder, f"visuals_{partial_no_S}_{partial_no_T}")

                                method.update_cfg(cfg.method)

                                # All methods other than Quantile
                                if cfg.method.cloud_from_path:

                                    registration_duration, resulting_transformation, additional_report = method.run(temp_src_path,
                                                                                                                    target_cloud_path,
                                                                                                                    visualization_path)

                                    if resulting_transformation is None:
                                        errors, success = {}, False
                                    else:
                                        # NOTE: Uses default parameters for registration distances etc.
                                        rotation, translation = qa.decompose_transformation(resulting_transformation)
                                        errors, success = qa.evaluate_registration(rotation, translation,
                                                                                   gt_rotation, gt_translation,
                                                                                   original_source, original_target, None)
                                # Quantile
                                else:

                                    downsampler = qa.downsampler_fcn[cfg.method.downsampler]
                                    source_cloud = downsampler(original_source, *downsampler_args)
                                    target_cloud = downsampler(original_target, *downsampler_args)
                                    registration_duration, resulting_transformation, metrics = method.run(source_cloud,
                                                                                                          target_cloud,
                                                                                                          visualization_path)

                                    *_, original_match_count, tuple_test_filtered, normal_tuple_test_filtered = metrics

                                    rotation, translation = qa.decompose_transformation(resulting_transformation)
                                    errors, success = qa.evaluate_registration(rotation, translation,
                                                                               gt_rotation, gt_translation,
                                                                               original_source, original_target, None)

                                    additional_report = f"Matches: {original_match_count} " \
                                                        f"(tuple filtered: {tuple_test_filtered}, normal filtered: {normal_tuple_test_filtered})\n"

                                report = custom_format(partial_no_S, partial_no_T, len(partial_numbers),
                                                       resulting_transformation, cfg.method.downsampler, down_param,
                                                       cfg.method.alpha, errors, success, registration_duration,
                                                       additional_report)

                                report_writer.write(report)
                                report_writer.flush()

                                total_success += success
                                total_pairs += 1

                    report_writer.write(f"Recall: {total_success / total_pairs}")