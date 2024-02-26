"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from common.registry import registry
import torch.distributed as dist
from datasets.builders.base_dataset_builder import BaseDatasetBuilder
from common.dist_utils import is_dist_avail_and_initialized
from datasets.datasets.arxiv_text_pair_datasets import ArxivTextPairDataset


@registry.register_builder("arxiv_caption")
class ArxivCaptionBuilder(BaseDatasetBuilder):
    DATASET_CONFIG_DICT = {
        "default": "train/pretrain_arxiv_stage1.yaml",
        "translator_train_stage2": "train/pretrain_arxiv_stage2.yaml",
        "translator_generate_stage1": "train/pretrain_arxiv_generate_stage1.yaml",
        "translator_generate_stage2": "train/pretrain_arxiv_generate_stage2.yaml"
    }

    def __init__(self, dataset_config, cfg):
        self.data_type = 'boi_feature'
        self.dataset_config = dataset_config
        self.runners_config = cfg.run_cfg
        self.args = cfg.args
        self.train_dataset_cls = ArxivTextPairDataset

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build(self):

        datasets = dict()
        split = "train"

        dataset_cls = self.train_dataset_cls

        datasets[split] = dataset_cls(
            cfg=self.dataset_config,
            mode='train'
        )

        return datasets
