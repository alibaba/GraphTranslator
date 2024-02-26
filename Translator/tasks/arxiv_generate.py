"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from common.logger import MetricLogger, SmoothedValue
from common.registry import registry
from tasks.base_task import BaseTask


@registry.register_task("arxiv_generate")
class ArxivGenerateTask(BaseTask):
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
        self.cfg = cfg
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

    def generate(
        self,
        iters_per_epoch,
        model,
        data_loader,
        scaler=None,
        dist_rank=None,
        log_freq=10,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None
        header = "Inference:"
        metric_logger = MetricLogger(delimiter="  ")
        if not hasattr(data_loader, "__next__"):
            data_loader = iter(data_loader)

        pred_txt = open(self.cfg.datasets_cfg['arxiv_caption']['pred_dir'], 'w')

        for network_input in metric_logger.log_every(data_loader, log_freq, iters_per_epoch, header):
            with torch.cuda.amp.autocast(enabled=use_amp):
                ChatGLM_response = model.generate(network_input, self.cfg.prompt_cfg['generate_prompt'])

            for i in range(len(ChatGLM_response)):
                id = str(network_input[0][i].detach().cpu().numpy())
                ori_desc = network_input[2][i].replace('\n', '\\n').replace('\t', '\\t')
                pred = ChatGLM_response[i].replace('\n', '\\n').replace('\t', '\\t')
                pred_txt.write(id+'\t'+ori_desc+'\t'+pred+'\n')

        pred_txt.close()
