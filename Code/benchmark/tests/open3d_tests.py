import numpy as np
import open3d

import Code.pipeline as pipeline
import Code.quantile_assignment as qa


def test_default_bunny():
    source_cloud_path = open3d.data.BunnyMesh().path
    target_cloud_path = source_cloud_path

    source_cloud = qa.read_cloud(source_cloud_path)
    target_cloud = qa.read_cloud(target_cloud_path)

    gt_indices = list(range(np.asarray(source_cloud.points).shape[0]))
    gt_cor = [gt_indices, gt_indices]

    default_rotation = [np.radians(60), np.pi / 2, np.pi]
    default_translation = [0.5, 0.5, 0.5]
    gt_rotation, gt_translation = qa.apply_transformation(source_cloud, default_rotation, default_translation)

    pipelineParameters = pipeline.PipelineParameters(source_cloud=source_cloud,
                                                     target_cloud=target_cloud,
                                                     downsampler="voxel",
                                                     downsampler_args=[0.01],
                                                     alpha=1.0,
                                                     tuple_test=False,
                                                     visualize=True,
                                                     refine_transformation_icp=True,
                                                     gt_trans=gt_translation,
                                                     gt_rot=gt_rotation,
                                                     gt_cor=gt_cor)

    transformation, metrics, success, *_ = pipeline.quantile_registration_pipeline(pipelineParameters)

    print(f"Resulting transformation is {transformation}")
    print(f"Metrics: {metrics}")
    print(f"Success: {success}")


def test_partial_bunny():
    source_cloud_path = open3d.data.BunnyMesh().path
    target_cloud_path = source_cloud_path

    source_cloud = qa.read_cloud(source_cloud_path)
    target_cloud = qa.read_cloud(target_cloud_path)

    # TODO: Indices for partial cases are now invalid
    # gt_indices = list(range(np.asarray(source_cloud.points).shape[0]))
    gt_cor = None

    source_cloud_cropped = source_cloud.crop(
        open3d.geometry.OrientedBoundingBox(center=np.array([0.0, 0.15, 0.0]), R=np.eye(3), extent=np.array([0.2] * 3)))
    target_cloud_cropped = target_cloud.crop(
        open3d.geometry.OrientedBoundingBox(center=np.array([0.0, -0.0, 0.0]), R=np.eye(3), extent=np.array([0.2] * 3)))

    theta = np.radians(60)
    default_rotation = [theta, theta * 2, np.pi]
    default_translation = [-0.2, 0.7, 0.5]

    gt_rotation, gt_translation = qa.apply_transformation(source_cloud_cropped, default_rotation, default_translation)

    pipelineParameters = pipeline.PipelineParameters(source_cloud=source_cloud_cropped,
                                                     target_cloud=target_cloud_cropped,
                                                     downsampler="uniform",
                                                     downsampler_args=[20],
                                                     alpha=0.5,
                                                     tuple_test=False,
                                                     visualize=True,
                                                     refine_transformation_icp=True,
                                                     gt_trans=gt_translation,
                                                     gt_rot=gt_rotation,
                                                     gt_cor=gt_cor)

    transformation, metrics, success, *_ = pipeline.quantile_registration_pipeline(pipelineParameters)

    print(f"Resulting transformation is {transformation}")
    print(f"Metrics: {metrics}")
    print(f"Success: {success}")


def test_default_armadillo():
    source_cloud_path = open3d.data.ArmadilloMesh().path
    target_cloud_path = source_cloud_path

    pipelineParameters = pipeline.PipelineParameters(source_cloud_path=source_cloud_path,
                                                     target_cloud_path=target_cloud_path,
                                                     downsampler="target",
                                                     downsampler_args=[1000],
                                                     alpha=1.0,
                                                     tuple_test=False,
                                                     visualize=True)

    transformation, *_ = pipeline.quantile_registration_pipeline(pipelineParameters)

    print(f"Resulting transformation is {transformation}")


if __name__ == "__main__":
    test_default_bunny()
    test_partial_bunny()
    test_default_armadillo()
