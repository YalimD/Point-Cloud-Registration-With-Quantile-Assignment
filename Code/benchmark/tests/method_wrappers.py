import copy
import os
import re
import subprocess
import time
from abc import abstractmethod

import numpy as np
from omegaconf import DictConfig

from Code.pipeline import quantile_registration_hydra


class RegistrationMethod:
    def __init__(self,
                 cfg: DictConfig):
        self.cfg = copy.deepcopy(cfg)

    # TODO: Add the return types to these guys
    @abstractmethod
    def run(self, source_cloud, target_cloud, visualization_path):
        pass

    def update_cfg(self, new_cfg: DictConfig):
        self.cfg = copy.deepcopy(new_cfg)


class QuantileAssignment(RegistrationMethod):
    def __init__(self,
                 cfg: DictConfig):
        RegistrationMethod.__init__(self, cfg)

        # For optimization, downsampling is expected to be applied beforehand
        self.cfg.downsampler = None

        # No init for quantile

    def update_cfg(self, new_cfg: DictConfig):
        super().update_cfg(new_cfg)
        self.cfg.downsampler = None

    # Note: As visualization is expected to change lot, parameter overrides the config
    def run(self, source_cloud, target_cloud, visualization_path):

        if self.cfg.visualize:
            self.cfg.visualize_path = visualization_path
            if not os.path.exists(visualization_path):
                os.mkdir(visualization_path)

        start = time.time()
        result = quantile_registration_hydra(self.cfg,
                                             source_cloud,
                                             target_cloud)
        duration = time.time() - start

        transformation, metrics = result[0], result[1:]

        return duration, transformation, metrics
