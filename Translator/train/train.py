import argparse
import random
import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import torch
import torch.backends.cudnn as cudnn

import tasks
from common.config import Config
from common.dist_utils import get_rank, init_distributed_mode
from common.logger import setup_logger
from common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from common.registry import registry
from common.utils import now

from datasets.builders import *
from models import *
from runners import *
from tasks import *

torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--cfg-path", default="./pretrain_arxiv_stage1.yaml", help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main(job_id):
    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    runner = get_runner_class(cfg)(
        cfg=cfg,
        job_id=job_id,
        task=task,
        model=model,
        datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    job_id = now()
    main(job_id)
