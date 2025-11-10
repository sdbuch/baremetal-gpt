from dataclasses import asdict
from datetime import datetime, timezone

import jax
import wandb
from omegaconf import DictConfig, OmegaConf

from bmgpt.config import Config, LoggerType


def get_logger_class_from_enum(logger_type: LoggerType):
    match logger_type:
        case LoggerType.PRINT:
            return PrintLogger
        case LoggerType.WANDB:
            return WandbLogger


def get_run_name(base: str) -> str:
    """Prepend current UTC time to config-specified run name"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{base}"
    return run_name


class Logger:
    def __init__(self, config: Config):
        self.project_name = config.project_name
        self.run_name = get_run_name(config.run_name)
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = asdict(config)
        self.config = config_dict

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def log(self, log_dict: dict):
        pass


class PrintLogger(Logger):
    def __init__(self, config: Config):
        super().__init__(config)

    def __enter__(self):
        print(f"Project: {self.project_name}")
        print(f"Run: {self.run_name}")
        return super().__enter__()

    def log(self, log_dict: dict):
        print(*[f"{metric}: {val}" for metric, val in log_dict.items()], sep="\t")


class WandbLogger(Logger):
    def __init__(self, config: Config):
        super().__init__(config)
        self.is_master = jax.process_index() == 0

    def __enter__(self):
        if self.is_master:
            wandb.init(
                project=self.project_name, name=self.run_name, config=self.config
            )
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_master:
            wandb.finish()
        return super().__exit__(exc_type, exc_value, traceback)

    def log(self, log_dict: dict):
        if self.is_master:
            wandb.log(log_dict)
