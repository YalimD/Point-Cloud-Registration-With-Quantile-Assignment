import logging

import tests
from Code.benchmark.tests import create_timed_folder
from tests.method_wrappers import *

import hydra
from omegaconf import DictConfig, OmegaConf

# Originally controlled by hydra
logger = logging.getLogger("root")

def handle_loggers(cfg):
    cfg.dataset.result_path = create_timed_folder(cfg.dataset.result_path)

    logger.handlers[0].setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(cfg.dataset.result_path, "process.log"), mode="w")
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)

    with open(os.path.join(cfg.dataset.result_path , "config_params.yaml"), "w") as cfg_writer:
        cfg_writer.write(OmegaConf.to_yaml(cfg))


@hydra.main(version_base=None,
            config_path="configs", config_name="edit_me.yaml")
def run_test(cfg: DictConfig) -> None:

    handle_loggers(cfg)

    method = None
    if cfg.method.name == "Quantile":
        method = QuantileAssignment(cfg.method)
    else:
        print(f"Invalid method {method}")

    if cfg.dataset.name == "3DMatch":
        tests.run_3dmatch(cfg, method)
    elif cfg.dataset.name == "Synthetic":
        tests.run_partial_matches(cfg, method)
    elif cfg.dataset.name == "KITTI":
        tests.run_kitti(cfg, method)
    else:
        print(f"Invalid dataset {cfg.dataset.name}")


if __name__ == "__main__":
    run_test()
