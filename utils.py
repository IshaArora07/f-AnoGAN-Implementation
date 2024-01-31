from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities.data import _flatten_dict
from torch.utils.tensorboard import SummaryWriter
import os
import random

from numbers import Number


from argparse import Namespace
from numbers import Number
from typing import Any, Dict

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(SummaryWriter):
    def __init__(
        self,
        log_dir: str = None,
        config: Namespace = None,
        enabled: bool = True,
        comment: str = '',
        purge_step: int = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = ''
    ):
        self.enabled = enabled
        if self.enabled:
            super().__init__(
                log_dir=log_dir,
                comment=comment,
                purge_step=purge_step,
                max_queue=max_queue,
                flush_secs=flush_secs,
                filename_suffix=filename_suffix
            )
        else:
            return

        # Add config
        if config is not None:
            self.add_hparams(
                {k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool, torch.Tensor))},
                {}
            )

    def log(self, data: Dict[str, Any], step: int) -> None:
        """Log each entry in data as its corresponding data type"""
        if self.enabled:
            for k, v in data.items():
                # Scalars
                if isinstance(v, Number):
                    self.add_scalar(k, v, step)

                # Images
                elif (isinstance(v, np.ndarray) or isinstance(v, torch.Tensor)) and len(v.shape) >= 3:
                    if len(v.shape) == 3:
                        self.add_image(k, v, step)
                    elif len(v.shape) == 4:
                        self.add_images(k, v, step)
                    else:
                        raise ValueError(f'Unsupported image shape: {v.shape}')

                else:
                    raise ValueError(f'Unsupported data type: {type(v)}')



def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class compute_auroc(Metric):
    """
    Computes the AUROC naively for each subgroup of the data individually.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)

        # Filter relevant subgroup
        subgroup_preds = preds[subgroups == subgroup]
        subgroup_targets = targets[subgroups == subgroup]

        # Compute the area under the ROC curve
        auroc = roc_auc_score(subgroup_targets, subgroup_preds)

        return torch.tensor(auroc, dtype=torch.float32)

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        auroc = roc_auc_score(targets, preds)
        return torch.tensor(auroc, dtype=torch.float32)

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_AUROC'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}AUROC'] = result
        return res   

class compute_average_precision(Metric):
    """
    Computes the Average Precision (AP) naively for each subgroup of the data individually.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True

    def __init__(self, subgroup_names: List[str]):
        super().__init__()
        self.subgroup_names = subgroup_names
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("subgroups", default=[], dist_reduce_fx=None)
        self.preds: List
        self.targets: List
        self.subgroups: List

    def update(self, subgroups: Tensor, preds: Tensor, targets: Tensor):
        """
        subgroups: Tensor of sub-group labels of shape [b]
        preds: Tensor of anomaly scores of shape [b]
        targets: Tensor of anomaly labels of shape [b]
        """
        assert len(preds) == len(targets) == len(subgroups)
        self.preds.append(preds)
        self.targets.append(targets)
        self.subgroups.append(subgroups)

    @property
    def num_subgroups(self):
        return len(self.subgroup_names)

    @staticmethod
    def compute_subgroup(preds: Tensor, targets: Tensor, subgroups: Tensor, subgroup: int):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)

        # Filter relevant subgroup
        subgroup_preds = preds[subgroups == subgroup]
        subgroup_targets = targets[subgroups == subgroup]

        # Compute the Average Precision
        ap = average_precision_score(subgroup_targets, subgroup_preds)

        return torch.tensor(ap, dtype=torch.float32)

    @staticmethod
    def compute_overall(preds: Tensor, targets: Tensor):
        if targets.sum() == 0 or targets.sum() == len(targets):
            return torch.tensor(0.)
        ap = average_precision_score(targets, preds)
        return torch.tensor(ap, dtype=torch.float32)

    def compute(self, **kwargs):
        preds = torch.cat(self.preds)  # [N]
        targets = torch.cat(self.targets)  # [N]
        subgroups = torch.cat(self.subgroups)  # [N]
        res = {}
        # Compute score for each subgroup
        for subgroup, subgroup_name in enumerate(self.subgroup_names):
            result = self.compute_subgroup(preds, targets, subgroups, subgroup)
            res[f'{subgroup_name}_AP'] = result
        # Compute score for whole dataset
        result = self.compute_overall(preds, targets)
        res[f'{common_string_left(self.subgroup_names)}AP'] = result
        return res
