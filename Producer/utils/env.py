import random
import os
import torch
import torch.distributed as dist
import numpy as np


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.distributed = True
        args.dist_backend = "nccl"

        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
        )
        torch.distributed.barrier()
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        print(
            "| distributed init (rank {}, world {}): {}".format(
                args.rank, args.world_size, args.dist_url
            ),
            flush=True,
        )
        setup_for_distributed(args.gpu == 0)
    else:
        print("Not using distributed mode")
        args.distributed = False
        return


def init_seeds(distributed, seed=0):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    random.seed(seed)
    np.random.seed(seed)

    if distributed:
        torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs.
    else:
        torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU.

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
