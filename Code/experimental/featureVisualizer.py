# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import platform
from Code.quantile_assignment import *

isMacOS = (platform.system() == "Darwin")


class Model:
    def __init__(self):
        self.name = None
        self.model_path = None
        self.original_model = None
        self.current_model = None
        self.current_features = None

        self.original_point_count = 0
        self.current_point_count = 0
        self.diameter = 0

        self.feature_colors = None


class Settings:
    UNLIT = "defaultUnlit"
    NORMALS = "normals"
    DEPTH = "depth"
    FEATURES = "defaultUnlit"

    DEFAULT_MATERIAL_NAME = "Default"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        }
    }

    DOWNSAMPLERS = {
        "Voxel": ("Voxel", downsample_voxel),
        "Uniform": ("Uniform", downsample_uniform),
        "Target": ("Target", downsample_to_target)
    }

    # Curvature is currently omitted
    FEATURE_TYPES = ["FPFH", "LFSH", "Curvature"]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.UNLIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = [1, 1, 1, 1]
        self.show_skybox = False
        self.show_axes = False

        # This is background (indirect background illumination) DON'T REMOVE
        self.use_ibl = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.UNLIT]

        self.downsample_method = self.DOWNSAMPLERS["Target"]
        self.downsample_param = 1000

        self.normal_neighbour_count = 30
        self.normal_neighbour_radius = 0.1

        self.current_feature_type = "FPFH"
        self.feature_neighbour_count = 30
        self.feature_neighbour_radius = 0.1

        # Models A and B
        self.models = [Model(), Model()]
        self.matrix = None
        self.max_affinity = 0
        self.global_max_affinity = 0

        self.selected_model_index = 0
        self.selected_model = None

        self.selected_point_index = 0
        self.selected_object_sphere = None
        self.normalize_colors = False


class AppWindow:
    MENU_OPEN_A = 1
    MENU_OPEN_B = 2
    MENU_EXPORT = 3
    MENU_QUIT = 4
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Unlit", "Normals", "Depth"]
    MATERIAL_SHADERS = [
        Settings.UNLIT, Settings.NORMALS, Settings.DEPTH
    ]

    # region Field setters

    # These methods require pressing "Update" to be applied

    def _normal_neighbour_change(self, normal_neighbour_count):
        self.settings.normal_neighbour_count = int(normal_neighbour_count)

    def _normal_radius_change(self, normal_neighbour_count):
        self.settings.normal_neighbour_radius = float(normal_neighbour_count)

    def _feature_neighbour_change(self, feature_neighbour_count):
        self.settings.feature_neighbour_count= int(feature_neighbour_count)

    def _feature_radius_change(self, feature_radius):
        self.settings.feature_neighbour_radius = float(feature_radius)

    def _on_feature_change(self, name, index):
        self.settings.current_feature_type = name

    def _on_downsample_type(self, method_name, method_index):
        self.settings.downsample_method = Settings.DOWNSAMPLERS[method_name]

    def _on_downsample_param(self, value):
        if self.settings.downsample_method[0] == "Voxel":
            self.settings.downsample_param = float(value)
        else:
            self.settings.downsample_param = int(value)

    # endregion

    # region recoloring

    def _on_model_update(self, value):

        self.settings.selected_model_index = int(value)
        self.settings.selected_model = self.settings.models[self.settings.selected_model_index]

        # Update the point index since it may be out of bounds according to other model
        self.settings.selected_point_index = min(self.settings.selected_point_index, self.settings.selected_model.current_point_count - 1)

        self.update_gui_elements()
        self._on_point_update(self.settings.selected_point_index)

    def _on_point_update_str(self, value):

        self._selected_point_index.int_value = int(value)
        self._on_point_update(min(self.settings.selected_model.current_point_count, int(value)))

    def _on_point_update(self, value):

        selected_model = self.settings.selected_model

        # Invalid index
        if value >= selected_model.current_point_count:
            return

        if selected_model.original_model is not None:

            self.settings.selected_point_index = int(value)
            self._recolor_models()

            # Color accordingly

            # Get the point coordinates and place a green ball on its location
            point_location = np.asarray(selected_model.current_model.points)[self.settings.selected_point_index]

            print(f"{value}: {point_location}")

            sphere_s = open3d.geometry.TriangleMesh.create_sphere(radius=self.settings.selected_model.diameter / 50)
            sphere_s.paint_uniform_color([0.0, 1.0, 0.0])
            sphere_s.translate(point_location)

            if self.settings.selected_object_sphere is not None:
                self._scene.scene.remove_geometry("sphere")

            self._scene.scene.add_geometry("sphere", sphere_s, rendering.MaterialRecord())
            self.settings.selected_object_sphere = sphere_s

    def _on_normalize_colors(self, normalize):
        self.settings.normalize_colors = normalize
        self._on_point_update(self.settings.selected_point_index)

    def _recolor_models(self):

        # Jet shaders
        def interpolate(value, y0, x0, y1, x1):
            if value < x0:
                return y0
            if value > x1:
                return y1
            return (value - x0) * (y1 - y0) / (x1 - x0) + y0

        def jet_shader(value):
            if value <= -0.75:
                return 0.0
            elif value <= -0.25:
                return interpolate(value, 0.0, -0.75, 1.0, -0.25)
            elif value <= 0.25:
                return 1.0
            elif value <= 0.75:
                return interpolate(value, 1.0, 0.25, 0.0, 0.75)
            return 0.0

        if len(list(filter(lambda m: m.current_point_count <= 1000, self.settings.models))) < 2:
            print("Please downsample point counts below 1000 before calculating features", file=sys.stderr)
            return

        if self.settings.matrix is None:
            print("Update the models first", file=sys.stderr)
            return

        # They should be swapped, since selected point is on the selected model but the other model is colored
        if self.settings.selected_model_index == 0:
            affinity_list = self.settings.matrix[int(self.settings.selected_point_index), :]
        else:
            affinity_list = self.settings.matrix[:, int(self.settings.selected_point_index)]

        self.settings.max_affinity = max(affinity_list)
        self._max_affinity.text = f"Max Affinity : {self.settings.max_affinity:.9f}"
        self._global_max_affinity.text = f"Global max Affinity : {self.settings.global_max_affinity:.9f}"

        # Normalization of current affinity values. If not, low mean affinities won't be visible
        if self.settings.normalize_colors:
            affinity_list = (affinity_list - min(affinity_list)) / (max(affinity_list) - min(affinity_list))

        selected_model = self.settings.selected_model
        other_model = self.settings.models[1 - self.settings.selected_model_index]

        assert(other_model.current_point_count == len(affinity_list))

        selected_model.current_model.paint_uniform_color(np.asarray([1.0, 1.0, 1.0]))

        colors = np.zeros((other_model.current_point_count, 3))

        for a, affinity in enumerate(affinity_list):
            value = affinity * 2 - 1
            color = np.asarray([jet_shader(value - 0.5), jet_shader(value), jet_shader(value + 0.5)])
            colors[a] = color

        other_model.current_model.paint_uniform_color(np.asarray([1.0, 1.0, 1.0]))
        np.asarray(other_model.current_model.colors)[:, :] = colors

        # Add the objects again to apply the coloring
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry(f"__model0__", self.settings.models[0].current_model, self.settings.material)
        self._scene.scene.add_geometry(f"__model1__", self.settings.models[1].current_model, self.settings.material)

    def update_gui_elements(self):

        if self.settings.models[0].original_model is not None and self.settings.models[1].original_model is not None:

            model_0, model_1 = self.settings.models

            self._current_point_count.text = f"Current Point Count 0:{model_0.current_point_count} - 1:{model_1.current_point_count}"
            self._original_point_count.text = f"Original Point Count 0:{model_0.original_point_count} - 1:{model_1.original_point_count}"
            self._object_diameter.text = f"Object Diameter: {self.settings.selected_model.diameter}"

            self._selected_point_index.set_limits(0, self.settings.selected_model.current_point_count - 1)
            self._selected_point_index.int_value = self.settings.selected_point_index

            self._selected_model_index.int_value = self.settings.selected_model_index
            self._selected_model_name.text = self.settings.selected_model.name

    # endregion

    # region GUIElements
    def feature_selection(self):
        feature_settings = gui.CollapsableVert("Feature settings", 0,
                                                gui.Margins(self.em, 0, 0, 0))
        feature_grid = gui.VGrid(2, 0.25 * self.em)

        self._features = gui.Combobox()
        for feature in Settings.FEATURE_TYPES:
            self._features.add_item(feature)
        self._features.set_on_selection_changed(self._on_feature_change)

        feature_grid.add_child(gui.Label("Visualize Type"))
        feature_grid.add_child(self._features)

        self._feature_neighbours = gui.TextEdit()
        self._feature_neighbours.text_value = str(self.settings.feature_neighbour_count)
        self._feature_neighbours.set_on_value_changed(self._feature_neighbour_change)

        feature_grid.add_child(gui.Label("Feature Neighbours"))
        feature_grid.add_child(self._feature_neighbours)

        self._feature_radius = gui.TextEdit()
        self._feature_radius.text_value = str(self.settings.feature_neighbour_radius)
        self._feature_radius.set_on_value_changed(self._feature_radius_change)

        feature_grid.add_child(gui.Label("Feature Radius"))
        feature_grid.add_child(self._feature_radius)

        feature_settings.add_child(feature_grid)

        self._settings_panel.add_child(feature_settings)
        self._settings_panel.add_fixed(self.separation_height)

    def model_point_selection(self):

        self._selected_model_index = gui.Slider(gui.Slider.INT)
        self._selected_model_index.set_limits(0, 1)
        self._selected_model_index.set_on_value_changed(self._on_model_update)

        self._selected_point_index = gui.Slider(gui.Slider.INT)
        self._selected_point_index.set_limits(0, 1)
        self._selected_point_index.set_on_value_changed(self._on_point_update)

        self._selected_point_index_str = gui.TextEdit()
        self._selected_point_index_str.set_on_value_changed(self._on_point_update_str)

        self._selected_model_name = gui.Label("Selected Model:")
        self._selected_model_name.text = f"{None}"

        grid = gui.VGrid(2, 0.25 * self.em)
        grid.add_child(gui.Label("Model"))
        grid.add_child(self._selected_model_index)
        grid.add_child(gui.Label("Point Index"))
        grid.add_child(self._selected_point_index)
        grid.add_child(gui.Label("Point Index"))
        grid.add_child(self._selected_point_index_str)
        grid.add_child(gui.Label("Selected Model:"))
        grid.add_child(self._selected_model_name)

        self._settings_panel.add_child(grid)

        self._normalize_colors = gui.Checkbox("Normalize Colors")
        self._normalize_colors.set_on_checked(self._on_normalize_colors)
        self._settings_panel.add_fixed(self.separation_height)
        self._settings_panel.add_child(self._normalize_colors)

    def downsample_selection(self):

        downsampler_settings = gui.CollapsableVert("Downsampler settings", 0,
                                                    gui.Margins(self.em, 0, 0, 0))

        self._current_downsampler = gui.Combobox()
        for prefab_name in sorted(Settings.DOWNSAMPLERS.keys()):
            self._current_downsampler.add_item(prefab_name)

        down_grid = gui.VGrid(2, 0.25 * self.em)
        down_grid.add_child(gui.Label("Method"))
        down_grid.add_child(self._current_downsampler)

        down_name, _ = self.settings.downsample_method
        self._current_downsampler.selected_text = down_name
        self._current_downsampler.set_on_selection_changed(self._on_downsample_type)

        self._downsample_parameter = gui.TextEdit()
        self._downsample_parameter.text_value = str(self.settings.downsample_param)
        self._downsample_parameter.set_on_value_changed(self._on_downsample_param)
        down_grid.add_child(gui.Label("Parameter"))
        down_grid.add_child(self._downsample_parameter)

        self._normal_neighbours = gui.TextEdit()
        self._normal_neighbours.text_value = str(self.settings.normal_neighbour_count)
        self._normal_neighbours.set_on_value_changed(self._normal_neighbour_change)
        down_grid.add_child(gui.Label("Normal Neighbours"))
        down_grid.add_child(self._normal_neighbours)

        self._normal_radius = gui.TextEdit()
        self._normal_radius.text_value = str(self.settings.normal_neighbour_radius)
        self._normal_radius.set_on_value_changed(self._normal_radius_change)
        down_grid.add_child(gui.Label("Normal Radius"))
        down_grid.add_child(self._normal_radius)

        downsampler_settings.add_child(down_grid)
        self._settings_panel.add_child(downsampler_settings)

        self._current_point_count = gui.Label("curCount")
        self._current_point_count.text = f"Current Point Count A:0 - B:0"
        self._settings_panel.add_child(self._current_point_count)

        self._original_point_count = gui.Label("orgCount")
        self._original_point_count.text = f"Original Point Count A:0 - B:0"
        self._settings_panel.add_child(self._original_point_count)

        self._object_diameter = gui.Label("objectDiameter")
        self._object_diameter.text = f"Object Diameter A:0 - B:0"
        self._settings_panel.add_child(self._object_diameter)

        self._max_affinity = gui.Label("maxAffinity")
        self._max_affinity.text = f"Max affinity: 0"
        self._settings_panel.add_child(self._max_affinity)

        self._global_max_affinity = gui.Label("globalMaxAffinity")
        self._global_max_affinity.text = f"Global Max affinity: 0"
        self._settings_panel.add_child(self._global_max_affinity)

        self._settings_panel.add_fixed(self.separation_height)

    # endregion

    # region modelUpdates

    def update_models(self):

        if len(list(filter(lambda m: m.original_model is not None, self.settings.models))) == 2:

            for m, model in enumerate(self.settings.models):
                # If the model is none, load the model first
                if model.original_model is None:
                    self.load(model.model_path, m)

                model.current_model = copy.deepcopy(model.original_model)
                model.current_model = self.settings.downsample_method[1](model.current_model,
                                                                         self.settings.downsample_param)
                model.current_model = calculate_normals(model.current_model,
                                                        self.settings.normal_neighbour_radius,
                                                        self.settings.normal_neighbour_count)

                model.current_point_count = np.asarray(model.current_model.points).shape[0]

                # Diameter according to original model
                model.diameter = np.linalg.norm(model.original_model.get_max_bound() - model.original_model.get_min_bound())

            feature_name = self.settings.current_feature_type

            for m, model in enumerate(self.settings.models):
                if feature_name == "FPFH":
                    model.current_features = calculate_fpfh(model.current_model,
                                                            self.settings.feature_neighbour_radius,
                                                            self.settings.feature_neighbour_count)
                elif feature_name == "Curvature":
                    model.current_features = calculate_cloud_curvatures(model.current_model,
                                                                        self.settings.feature_neighbour_radius,
                                                                        self.settings.feature_neighbour_count)
                elif feature_name == "LFSH":
                    model.current_features = calculate_lfsh(model.current_model,
                                                            self.settings.feature_neighbour_radius,
                                                            self.settings.feature_neighbour_count)

            model_0, model_1 = self.settings.models

            # TEST: Use position as a feature to debug the coloring
            # model_0.current_features = np.array(model_0.current_model.points)
            # model_1.current_features = np.array(model_1.current_model.points) - np.asarray([model_1.diameter, 0.0, 0.0])

            self.settings.matrix = calculate_affinity(model_0.current_features,
                                                      model_1.current_features, 1, False)

            self.settings.global_max_affinity = np.max(self.settings.matrix)

            # Calculate the current features and color the model
            self.update_gui_elements()
            self._on_model_update(self.settings.selected_model_index)
            self._apply_settings()
        else:
            print("Models are not loaded, ignoring", file=sys.stderr)
            return

    def load(self, path, index):
        self._scene.scene.clear_geometry()

        model = self.settings.models[index]

        model.name = os.path.basename(path)
        model.model_path = path
        model.original_model = None

        cloud = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)
            cloud = open3d.geometry.PointCloud(points=mesh.meshes[0].mesh.vertices)
        else:
            print("[Info]", path, "appears to be a point cloud")
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass

        if cloud is not None:
            print("[Info] Successfully read", path)
            if not cloud.has_normals():
                cloud.estimate_normals()
            cloud.normalize_normals()
        else:
            print("[WARNING] Failed to read points", path)

        if cloud is not None:
            try:
                # Point cloud
                model = self.settings.models[index]

                model.diameter = np.linalg.norm(cloud.get_max_bound() - cloud.get_min_bound())

                # Move the model a bit
                if index == 1:
                    cloud.translate(np.asarray([model.diameter, 0.0, 0.0]))

                self._scene.scene.add_geometry(f"__model{index}__", cloud,
                                               self.settings.material)
                model.original_model = cloud
                model.current_model = copy.deepcopy(model.original_model)
                model.original_point_count = np.asarray(cloud.points).shape[0]
                model.current_point_count = model.original_point_count

                if self.settings.models[1 - index].original_model is not None:
                    # Add the other model back, since managing the models is buggy
                    self._scene.scene.add_geometry(f"__model{1 - index}__",
                                                   self.settings.models[1 - index].original_model,
                                                   self.settings.material)

                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(45, bounds, bounds.get_center())

                self._on_model_update(int(index))

            except Exception as e:
                print(e)

    # endregion

    # region Don't touch

    def __init__(self, width, height):
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + AppWindow.DEFAULT_IBL

        self.window = gui.Application.instance.create_window(
            "Feature Visualizer", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self.em = em
        self.separation_height = separation_height

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert,
        # gui.CollapsableVert, and gui.VGrid. By nesting the layouts we can
        # achieve complex designs. Usually we use a vertical layout as the
        # topmost widget, since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._arcball_button = gui.Button("Arcball")
        self._arcball_button.horizontal_padding_em = 0.5
        self._arcball_button.vertical_padding_em = 0
        self._arcball_button.set_on_clicked(self._set_mouse_mode_rotate)
        self._model_button = gui.Button("Model")
        self._model_button.horizontal_padding_em = 0.5
        self._model_button.vertical_padding_em = 0
        self._model_button.set_on_clicked(self._set_mouse_mode_model)
        view_ctrls.add_child(gui.Label("Mouse controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._arcball_button)
        h.add_child(self._model_button)
        h.add_stretch()
        view_ctrls.add_child(h)

        self._show_skybox = gui.Checkbox("Show skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_skybox)

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)
        view_ctrls.add_fixed(separation_height)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))

        self._shader = gui.Combobox()
        for item in AppWindow.MATERIAL_NAMES:
            self._shader.add_item(item)
        self._shader.set_on_selection_changed(self._on_shader)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)

        self.downsample_selection()
        self.feature_selection()
        self.model_point_selection()

        self._update_buttom = gui.Button("Update")
        self._update_buttom.set_on_clicked(self.update_models)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(self._update_buttom)

        # ----

        # Normally our user interface can be children of all one layout (usually
        # a vertical layout), which is then the only child of the window. In our
        # case we want the scene to take up all the space and the settings panel
        # to go above it. We can do this custom layout by providing an on_layout
        # callback. The on_layout callback should set the frame
        # (position + size) of every child correctly. After the callback is
        # done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", AppWindow.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open Model A", AppWindow.MENU_OPEN_A)
            file_menu.add_item("Open Model B", AppWindow.MENU_OPEN_B)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN_A, self._on_menu_open_A)
        w.set_on_menu_item_activated(AppWindow.MENU_OPEN_B, self._on_menu_open_B)
        w.set_on_menu_item_activated(AppWindow.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(AppWindow.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # ----

        self.load("..\\..\\Data\\Dummy\\bunny\\bunny.ply", 0)
        self.load("..\\..\\Data\\Dummy\\bunny\\bunny.ply", 1)

        self._apply_settings()

    def _apply_settings(self):
        self._scene.scene.set_background(self.settings.bg_color)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_axes(self.settings.show_axes)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._point_size.double_value = self.settings.material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_use_ibl(self, use):
        self.settings.use_ibl = use
        self._apply_settings()

    def _on_shader(self, name, index):
        self.settings.set_material(AppWindow.MATERIAL_SHADERS[index])
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def menu_open_helper(self, model_name):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, f"Choose file to load for {model_name}",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)

        return dlg

    def _on_menu_open_A(self):
        dlg = self.menu_open_helper("Model A")
        dlg.set_on_done(self._on_load_dialog_done_A)
        self.window.show_dialog(dlg)

    def _on_menu_open_B(self):
        dlg = self.menu_open_helper("Model B")
        dlg.set_on_done(self._on_load_dialog_done_B)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done_A(self, filename):
        self.window.close_dialog()
        self.load(filename, 0)

    def _on_load_dialog_done_B(self, filename):
        self.window.close_dialog()
        self.load(filename, 1)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            AppWindow.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    # endregion

def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path, 0)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()


if __name__ == "__main__":
    main()