"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset, IterableDataset
from torch.utils.data.dataloader import default_collate


class BatchIterableDataset(IterableDataset):
    def __init__(self, cfg, mode):
        super(BatchIterableDataset, self).__init__()
        self._cfg = cfg
        self._mode = mode

        self.summary_embeddings = pd.read_csv(cfg['datasets_dir'])
        self.row_count = self.summary_embeddings.shape[0]
        self.start_pos = 0
        self.end_pos = self.summary_embeddings.shape[0]

        self._parser_dict = {
            "train": self._train_data_parser,
            "eval": self._eval_data_parser,
            "infer": self._infer_data_parser
        }

    def _train_data_parser(self, data):
        raise NotImplementedError

    def _eval_data_parser(self, data):
        raise NotImplementedError

    def _infer_data_parser(self, data):
        raise NotImplementedError

    def data_iterator(self):
        for _, row in self.summary_embeddings.iterrows():
            try:
                data = [tuple(row)]
            except Exception:
                break
            yield self._parser_dict[self._mode](data)

    def __iter__(self):
        return self.data_iterator()


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
