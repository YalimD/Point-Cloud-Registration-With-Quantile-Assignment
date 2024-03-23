import hydra
from omegaconf import DictConfig

from dataclasses import dataclass, field

from Code.quantile_assignment import *


@dataclass
class PipelineParameters:
    source_cloud: open3d.geometry.PointCloud = None
    source_cloud_path: str = ""

    target_cloud: open3d.geometry.PointCloud = None
    target_cloud_path: str = ""

    downsampler: str = "uniform"
    downsampler_args: List = field(default_factory=lambda: [10])

    calculate_normals: bool = True
    normal_knn_radius: float = 0.1
    normal_knn_max_nn: int = 30

    use_fpfh: bool = True
    use_lfsh: bool = False
    feature_nn_radius: float = 0.5
    feature_nn_count: int = 100

    # False is FGR
    use_quantile_matching: bool = True
    matching_method: str = "hungarian_cost"
    affinity_penalty_coefficient: float = 1.0
    save_affinity: bool = False
    alpha: float = -1

    tuple_test: bool = True  # FGR based tuple test, not aggressive
    tuple_test_maximum_pair_count: int = 1000
    tuple_test_tau: float = 0.9
    tuple_test_seed: int = 0

    transformation_calculator: str = "fgr"
    refine_transformation_icp: bool = True
    icp_distance_threshold: float = 0.01

    tuple_normal_alignment_check: bool = False
    tuple_normal_alignment_threshold_degrees: float = 5

    visualize: bool = False
    visualize_path: str = "."
    parameter_dump_file: str = ".\\parameters.txt"

    gt_rot: np.ndarray = None
    gt_trans: np.ndarray = None
    gt_cor: List = None

    projection_error_threshold_meters: float = 0.2
    rotation_threshold_degrees: float = 5.0
    translation_threshold_meters: float = 2.0


def quantile_registration_pipeline(parameters: PipelineParameters):

    parameter_dict = dir(parameters)
    with open(parameters.parameter_dump_file, "w") as parameter_writer:
        for key in parameter_dict:
            value = getattr(parameters, key)
            if not str(key).startswith("__"):
                if str(key) == "gt_cor" or type(value) == open3d.geometry.PointCloud:
                    parameter_writer.write(f"Key: {key} - value: GIVEN\n")
                else:
                    parameter_writer.write(f"Key: {key} - value: {value}\n")

    source_cloud = parameters.source_cloud
    if source_cloud is None:
        source_cloud = read_cloud(parameters.source_cloud_path)

    target_cloud = parameters.target_cloud
    if target_cloud is None:
        target_cloud = read_cloud(parameters.target_cloud_path)

    if source_cloud is None or target_cloud is None:
        return None

    original_source = copy.deepcopy(source_cloud)
    original_target = copy.deepcopy(target_cloud)

    original_match_count = 0
    tuple_test_filtered = 0
    normal_tuple_test_filtered = 0

    if parameters.downsampler is not None and parameters.downsampler_args is not None:
        source_cloud = downsampler_fcn[parameters.downsampler](source_cloud, *parameters.downsampler_args)
        target_cloud = downsampler_fcn[parameters.downsampler](target_cloud, *parameters.downsampler_args)

    if parameters.calculate_normals:
        source_cloud = calculate_normals(source_cloud,
                                         parameters.normal_knn_radius,
                                         parameters.normal_knn_max_nn)
        target_cloud = calculate_normals(target_cloud,
                                         parameters.normal_knn_radius,
                                         parameters.normal_knn_max_nn)

    if parameters.use_fpfh:
        source_features = calculate_fpfh(source_cloud,
                                         parameters.feature_nn_radius,
                                         parameters.feature_nn_count)
        target_features = calculate_fpfh(target_cloud,
                                         parameters.feature_nn_radius,
                                         parameters.feature_nn_count)
    elif parameters.use_lfsh:
        logging.info("Using lfsh")

        source_indices = np.arange(np.asarray(source_cloud.points).shape[0])
        source_diameter = np.linalg.norm(source_cloud.get_max_bound() - source_cloud.get_min_bound())
        source_sup_radius = source_diameter * 0.13
        source_features = calculate_lfsh(source_cloud, source_indices, source_sup_radius, 30)

        target_indices = np.arange(np.asarray(target_cloud.points).shape[0])
        target_diameter = np.linalg.norm(target_cloud.get_max_bound() - target_cloud.get_min_bound())
        target_sup_radius = target_diameter * 0.13
        target_features = calculate_lfsh(target_cloud, target_indices, target_sup_radius, 30)

    else:
        source_features = calculate_cloud_curvatures(source_cloud,
                                                     parameters.feature_nn_radius,
                                                     parameters.feature_nn_count)
        target_features = calculate_cloud_curvatures(target_cloud,
                                                     parameters.feature_nn_radius,
                                                     parameters.feature_nn_count)

        remove_nan_values(np.asarray(source_cloud.points))
        remove_nan_values(np.asarray(target_cloud.points))

    if parameters.visualize:
        visualize_clouds(source_cloud, target_cloud, None, parameters.visualize_path, "preview")

    if parameters.use_quantile_matching:

        logging.info("Using the quantile method")

        result = quantile_registration(source_features,
                                       target_features,
                                       parameters.affinity_penalty_coefficient,
                                       parameters.save_affinity,
                                       parameters.alpha,
                                       matching_lib[parameters.matching_method])

        matches, weights, alpha, k_alpha, best_q, cost = result

        if parameters.alpha == -1:
            logging.info(f"Using alpha {alpha}")

        logging.info(f"Best q: {best_q}. Cost: {cost}")

        original_match_count = len(matches[0])

        if parameters.tuple_test:
            filtered_indices = fgr_tuple_test(source_cloud, target_cloud,
                                              matches,
                                              parameters.tuple_test_maximum_pair_count,
                                              parameters.tuple_test_tau,
                                              parameters.tuple_test_seed, normal_alignment=False)

            matches = np.asarray([matches[0][filtered_indices], matches[1][filtered_indices]])
            weights = np.asarray(weights[filtered_indices])

            tuple_test_filtered = original_match_count - len(matches[0])
    else:

        logging.info("Using the FGR method")

        wrapped_source_features = open3d.pipelines.registration.Feature()
        wrapped_source_features.data = source_features.T

        wrapped_target_features = open3d.pipelines.registration.Feature()
        wrapped_target_features.data = target_features.T

        result = open3d.pipelines.registration.registration_fgr_based_on_feature_matching(source_cloud,
                                                                                          target_cloud,
                                                                                          wrapped_source_features,
                                                                                          wrapped_target_features,
                                                                                          open3d.pipelines.registration.FastGlobalRegistrationOption(
                                                                                              tuple_test=False))

        matches = np.asarray(result.correspondence_set).T
        weights = np.ones_like((len(matches[0])))

        original_match_count = len(matches[0])

    rotation, translation = transformation_fnc[parameters.transformation_calculator](source_cloud, target_cloud,
                                                                                     matches, weights)

    if parameters.tuple_normal_alignment_check:
        pre_tuple_count = len(matches[0])
        source_cloud_copy = copy.deepcopy(source_cloud)
        target_cloud_copy = copy.deepcopy(target_cloud)

        source_cloud_copy.rotate(rotation, center=(0, 0, 0)).translate(translation)

        filtered_indices = fgr_tuple_test(source_cloud_copy, target_cloud_copy,
                                          matches,
                                          parameters.tuple_test_maximum_pair_count,
                                          parameters.tuple_test_tau,
                                          parameters.tuple_test_seed,
                                          normal_alignment=True,
                                          normal_alignment_threshold_degree=parameters.tuple_normal_alignment_threshold_degrees,
                                          visualize=False)

        matches = np.asarray([matches[0][filtered_indices], matches[1][filtered_indices]])
        weights = np.asarray(weights[filtered_indices])

        rotation, translation = transformation_fnc[parameters.transformation_calculator](source_cloud, target_cloud,
                                                                                         matches, weights)

        normal_tuple_test_filtered = pre_tuple_count - len(matches[0])

    if parameters.refine_transformation_icp:
        transformation_matrix = calculate_transformation_matrix(rotation, translation)

        rotation, translation = refine_transformation_icp(source_cloud, target_cloud,
                                                          transformation_matrix,
                                                          parameters.icp_distance_threshold)

    # Show the original placement, matches then transformation
    if parameters.visualize:
        # Visualize the top 10 matches lines
        top_matches_index = min(10, len(matches[0]))
        top_matches = matches[0][:top_matches_index], matches[1][:top_matches_index]

        visualize_clouds(source_cloud, target_cloud, top_matches, parameters.visualize_path, "lines")

        visualize_transformation(source_cloud, target_cloud, rotation, translation, parameters.visualize_path)

    metrics, success = evaluate_registration(rotation, translation,
                                             parameters.gt_rot, parameters.gt_trans,
                                             original_source, original_target,
                                             parameters.gt_cor,
                                             parameters.projection_error_threshold_meters,
                                             parameters.rotation_threshold_degrees,
                                             parameters.translation_threshold_meters)

    # Return the transformed original source cloud as well
    # TODO: Isn't this suppose to be the original
    transformed_source = source_cloud.rotate(rotation, center=(0, 0, 0)).translate(translation)
    return calculate_transformation_matrix(rotation, translation), \
           metrics, success, \
           transformed_source, target_cloud, \
           original_match_count, tuple_test_filtered, normal_tuple_test_filtered


# Source and target cloud will be reused if given, to save space and time
def quantile_registration_hydra(cfg: DictConfig,
                                source_cloud: None,
                                target_cloud: None):

    # Map the configuration to the parameters
    parameters = PipelineParameters()

    for item in cfg.items():
        setattr(parameters, item[0], item[1] if item[1] != 'None' else None)

    if source_cloud is not None:
        parameters.source_cloud = source_cloud

    if target_cloud is not None:
        parameters.target_cloud = target_cloud

    return quantile_registration_pipeline(parameters)


@hydra.main(version_base=None,
            config_path="benchmark/configs/method", config_name="quantile_default.yaml")
def run_quantile_pipeline(cfg: DictConfig) -> None:

    quantile_registration_hydra(cfg, None, None)


if __name__ == "__main__":
    run_quantile_pipeline()
