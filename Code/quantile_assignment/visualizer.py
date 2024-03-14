import copy
import os.path

import numpy as np
import open3d


def normalize_colors(cloud, output_channel=0):
    colors = abs(np.asarray(cloud.colors)[:, 0])

    col_std = np.std(colors)
    col_mean = np.average(colors)

    max_color = col_mean + 2 * col_std
    min_color = col_mean - 2 * col_std

    colors = (colors - min_color) / (max_color - min_color)
    np.asarray(cloud.colors)[:, output_channel] = colors


def calculate_correspondence_lines(matches,
                                   sphere_radius: float = 0.01,
                                   line_color=np.array([0.0, 1.0, 0.0])) -> open3d.geometry.LineSet:
    points = np.concatenate((matches[0], matches[1]), axis=0)

    colors = [line_color for i in range(len(matches[0]))]
    lines = []
    point_spheres = []

    for i in range(len(matches[0])):
        lines.append([i, i + len(matches[0])])

        sphere_s = open3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere_s.paint_uniform_color(line_color)
        sphere_s.translate(points[i])
        point_spheres.append(sphere_s)

        sphere_t = open3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere_t.paint_uniform_color(line_color)
        sphere_t.translate(points[i + len(matches[0])])
        point_spheres.append(sphere_t)

    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)

    return line_set, point_spheres


def visualize_clouds(source_cloud: open3d.geometry.PointCloud,
                     target_cloud: open3d.geometry.PointCloud,
                     matches=None,
                     output_path=".",
                     output_file_name="output") -> None:
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    source_cloud_copy = copy.deepcopy(source_cloud)
    target_cloud_copy = copy.deepcopy(target_cloud)

    drawables = []

    # Draw Matches
    if matches:
        point_radius = np.linalg.norm(source_cloud.get_max_bound() - source_cloud.get_min_bound()) / 50
        line_set, point_spheres = calculate_correspondence_lines([np.asarray(source_cloud_copy.points)[matches[0]],
                                                                  np.asarray(target_cloud_copy.points)[matches[1]]],
                                                                 point_radius)
        drawables.append(line_set)
        for sphere in point_spheres:
            drawables.append(sphere)

    # Curvatures are stored as colors
    if source_cloud_copy.has_colors():
        normalize_colors(source_cloud_copy, 0)
    else:
        source_cloud_copy.paint_uniform_color(np.asarray([1.0, 0.0, 0.0]))

    if target_cloud_copy.has_colors():
        normalize_colors(target_cloud_copy, 2)
    else:
        target_cloud_copy.paint_uniform_color(np.asarray([0.0, 0.0, 1.0]))

    drawables.extend([source_cloud_copy, target_cloud_copy])

    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window(width=720, height=720, visible=False)
    for drawable in drawables:
        visualizer.add_geometry(drawable)
    visualizer.update_renderer()

    rad_per_pixel = 0.003
    min_point = np.min(np.vstack((source_cloud_copy.get_min_bound(), target_cloud_copy.get_min_bound())), axis=0)
    max_point = np.max(np.vstack((source_cloud_copy.get_max_bound(), target_cloud_copy.get_max_bound())), axis=0)
    scene_center = (max_point + min_point) / 2

    view_control = visualizer.get_view_control()
    cam_og = copy.deepcopy(view_control.convert_to_pinhole_camera_parameters())
    cam_location = cam_og.extrinsic[:3, 3]
    cam_radius = np.linalg.norm(cam_location - scene_center)
    scene_radius = np.linalg.norm(max_point - min_point) / 1.8

    old_radius = cam_radius
    cam_radius = scene_radius * np.sqrt(3)
    view_control.camera_local_translate(-(cam_radius - old_radius), 0, 0)
    visualizer.update_renderer()

    cam_og = copy.deepcopy(view_control.convert_to_pinhole_camera_parameters())

    sides = ["front", "right", "back", "left", "top", "bottom"]
    for side in sides:
        if side == "right" or side == "back" or side == "left":
            view_control.camera_local_translate(cam_radius, cam_radius, 0)
            view_control.camera_local_rotate(np.deg2rad(-90) / rad_per_pixel, 0)
        elif side == "top" or side == "bottom":
            view_control.convert_from_pinhole_camera_parameters(cam_og)
            direction = 1 if side == "top" else -1
            view_control.camera_local_translate(cam_radius, 0, cam_radius * direction)
            view_control.camera_local_rotate(0, np.deg2rad(90 * direction) / rad_per_pixel)
        visualizer.update_renderer()
        visualizer.capture_screen_image(os.path.join(output_path, f"{output_file_name}_{side}.png"), True)


def visualize_transformation(source_cloud, target_clouds, rotation, translation, output_path=".") -> None:
    source_cloud_transformed = copy.deepcopy(source_cloud).rotate(rotation, center=(0, 0, 0)).translate(translation)

    visualize_clouds(source_cloud_transformed, target_clouds, None, output_path, "registered")
