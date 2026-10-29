# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import math
import os
import pathlib
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, List, Optional

from omegaconf import DictConfig, OmegaConf

import echoheart.distributed as distributed
from echoheart.logging import cleanup_logging, setup_logging
from echoheart.utils import fix_random_seeds

logger = logging.getLogger("echoheart")


@dataclass
class PretrainingSetupArgs:
    config_file: str
    output_dir: str = ""
    opts: List[Any] = field(default_factory=lambda: [])

    def __post_init__(self):
        # Convert configuration sequences to ordinary lists for CLI overrides.
        if OmegaConf.is_config(self.opts):
            self.opts = OmegaConf.to_object(self.opts)


def apply_scaling_rules_to_cfg(cfg):  # to fix
    assert distributed.is_enabled(), "Setup distributed to get global size !"
    if "schedules" in cfg:
        # For schedules v2, the scaling rules are applied when building the schedules, the config is not modified
        return cfg

    if cfg.optim.scaling_rule == "linear_wrt_256":
        old_lr = cfg.optim.lr
        cfg.optim.lr *= cfg.train.batch_size_per_gpu * distributed.get_world_size() / 256.0
        logger.info(f"linear scaling learning rate; old: {old_lr}, new: {cfg.optim.lr}")
    elif cfg.optim.scaling_rule == "sqrt_wrt_1024":
        old_lr = cfg.optim.lr
        cfg.optim.lr *= 4 * math.sqrt(cfg.train.batch_size_per_gpu * distributed.get_world_size() / 1024.0)
        logger.info(f"sqrt scaling learning rate; old: {old_lr}, new: {cfg.optim.lr}")
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    output_dir = os.path.abspath(output_dir)
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_default_config() -> DictConfig:
    p = pathlib.Path(__file__).parent / "ssl_default_config.yaml"
    return OmegaConf.load(p)


def get_cfg_from_args(args: PretrainingSetupArgs, strict=True):
    overrides = [*args.opts]
    if args.output_dir is not None:
        overrides.append(f"train.output_dir={os.path.realpath(args.output_dir)}")

    # Config file
    cfg = OmegaConf.load(args.config_file)

    # Command line overrides
    opts_cfg = OmegaConf.from_cli(overrides)

    # Default config
    default_cfg = get_default_config()
    if strict:
        OmegaConf.set_struct(default_cfg, True)
    cfg = OmegaConf.merge(default_cfg, cfg, opts_cfg)
    return cfg


def setup_config(args: PretrainingSetupArgs, strict_cfg=True):
    """
    Create configs and perform basic setups.
    """
    # Create the cfg with OmegaConf
    cfg = get_cfg_from_args(args, strict=strict_cfg)
    # setup distributed, logging, and random seeds
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    # dump config before modifying so it can be reloaded
    if args.output_dir is not None:
        write_config(cfg, args.output_dir)
    # modify the config inplace by applying scaling rules
    apply_scaling_rules_to_cfg(cfg)
    return cfg


def setup_job(
    output_dir: Optional[str] = None,
    distributed_enabled: bool = True,
    logging_enabled: bool = True,
    seed: Optional[int] = 0,
    restrict_print_to_main_process: bool = True,
    distributed_timeout: timedelta | None = None,
):
    """
    Setup methods shared by EchoHeart training jobs.
    Initializes logging, distributed, random seeds and other utilities.
    """
    if output_dir is not None:
        output_dir = os.path.realpath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if logging_enabled:
        setup_logging(
            output=output_dir,
            level=logging.INFO,
            log_to_stdout_only_in_main_process=restrict_print_to_main_process,
        )

    if distributed_enabled:
        distributed.enable(
            overwrite=True,
            nccl_async_error_handling=True,
            restrict_print_to_main_process=restrict_print_to_main_process,
            timeout=distributed_timeout,
        )

    if seed is not None:
        rank = distributed.get_rank()
        fix_random_seeds(seed + rank)

def exit_job(distributed_enabled: bool = True, logging_enabled: bool = True):
    if distributed_enabled:
        distributed.disable()
    if logging_enabled:
        cleanup_logging()
