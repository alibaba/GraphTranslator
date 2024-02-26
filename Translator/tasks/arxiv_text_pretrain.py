"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from common.registry import registry
from tasks.base_task import BaseTask


@registry.register_task("arxiv_text_pretrain")
class ArxivTextPretrainTask(BaseTask):

    def __init__(self, **kwargs):
        super().__init__()

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        assert len(cfg.datasets_cfg) > 0, "At least one dataset has to be specified."

        for name in cfg.datasets_cfg:
            dataset_config = cfg.datasets_cfg[name]
            builder = registry.get_builder_class(name)(dataset_config, cfg)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass
