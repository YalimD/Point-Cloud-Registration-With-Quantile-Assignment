import numpy as np


# NOTE: Creates a new point for each voxel via averaging existing points
def downsample_voxel(cloud, voxel_size=0.1):
    return cloud.voxel_down_sample(voxel_size=voxel_size)


def downsample_uniform(cloud, sample_step=10):
    return cloud.uniform_down_sample(sample_step)


def downsample_to_target(cloud, target_point_count):
    sample_step = int(np.ceil(len(np.asarray(cloud.points)) / target_point_count))
    if sample_step <= 1:
        return cloud
    return cloud.uniform_down_sample(sample_step)


downsampler_fcn = {
    "voxel": downsample_voxel,
    "uniform": downsample_uniform,
    "target": downsample_to_target
}
