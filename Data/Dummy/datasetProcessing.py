
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import glob

import os, numpy as np


data_path = "Datasets\\Synthetic\\Original"
the_file_list = glob.glob(f"{data_path}\\**\\*.ply", recursive=True)

for cloud_path in tqdm(the_file_list):

    print(f"Processing {cloud_path}")
    cloud_path = os.path.abspath(cloud_path)
    cloud_float_path = f"{os.path.splitext(cloud_path)[0]}_float.ply"

    if not os.path.exists(cloud_float_path):
        model = PlyData.read(cloud_path)
        vertices = np.array(list(map(lambda element: tuple([element[0], element[1], element[2]]), model["vertex"])),
                            dtype=[("x", "float32"), ("y", "float32"), ("z", "float32")])
        PlyData([PlyElement.describe(vertices, "vertex")], text=True).write(cloud_float_path)
        model = None

    os.remove(cloud_path)
    os.rename(cloud_float_path, cloud_path)