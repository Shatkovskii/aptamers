import os
from pathlib import Path

import dotenv
from hydra import compose, initialize
from omegaconf import OmegaConf

dotenv.load_dotenv(".env")

CONFIG_PATH = os.environ["CONFIG_PATH"]


def get_config(
    overrides: list[str] | None = None,
    config_path: Path = Path("conf"),
    config_name: str = "config"
) -> dict:
    with initialize(version_base=None, config_path=str(config_path)):
        hydra_config = compose(config_name=config_name, overrides=overrides)
        return OmegaConf.to_object(hydra_config)
