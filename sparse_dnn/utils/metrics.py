from typing import Any, Dict, Optional

import torch
import numpy as np
from torch import Tensor
from timm.utils.metrics import AverageMeter as TimmAverageMeter

class ValueEpoch:
    def __init__(self, val: Any, epoch: int):
        self.val = val
        self.epoch = epoch

    def load_state_dict(self, state_dict: Dict[str, Tensor], prefix: str = ''):
        if len(prefix) > 0 and not prefix.endswith('.'):
            prefix += '.'
        self.val = state_dict[f'{prefix}val'].item()
        self.epoch = int(state_dict[f'{prefix}epoch'].item())

    def state_dict(self, state_dict: Dict[str, Tensor] = None, prefix: str = '') -> Dict[str, Tensor]:
        if state_dict is None:
            state_dict = {}
        if len(prefix) > 0 and not prefix.endswith('.'):
            prefix += '.'

        val = self.val.cpu() if isinstance(self.val, Tensor) else torch.tensor(self.val, device='cpu')
        state_dict[f'{prefix}val'] = val
        state_dict[f'{prefix}epoch'] = torch.tensor(self.epoch, device='cpu')
        return state_dict

class AverageMeter(TimmAverageMeter):
    def load_state_dict(self, state_dict: Dict[str, Tensor], prefix: str = '') -> None:
        if len(prefix) > 0 and not prefix.endswith('.'):
            prefix += '.'
        self.val = state_dict[f'{prefix}val'].item()
        self.avg = state_dict[f'{prefix}avg'].item()
        self.sum = state_dict[f'{prefix}sum'].item()
        self.count = state_dict[f'{prefix}count'].item()

    def state_dict(self, state_dict: Dict[str, Tensor] = None, prefix: str = '') -> Dict[str, Tensor]:
        if state_dict is None:
            state_dict = {}
        if len(prefix) > 0 and not prefix.endswith('.'):
            prefix += '.'
        state_dict[f'{prefix}val'] = torch.tensor(self.val, device='cpu')
        state_dict[f'{prefix}avg'] = torch.tensor(self.avg, device='cpu')
        state_dict[f'{prefix}sum'] = torch.tensor(self.sum, device='cpu')
        state_dict[f'{prefix}count'] = torch.tensor(self.count, device='cpu')
        return state_dict

class AverageMeterWarmup(AverageMeter):
    def __init__(self, warmup: int) -> None:
        self.warmup = warmup
        self.total_count = 0
        super().__init__()

    def load_state_dict(self, state_dict: Dict[str, Tensor], prefix: str = '') -> None:
        if len(prefix) > 0 and not prefix.endswith('.'):
            prefix += '.'
        self.warmup = int(state_dict[f'{prefix}warmup'].item())
        self.total_count = int(state_dict[f'{prefix}total_count'].item())
        super().load_state_dict(state_dict, prefix)

    def state_dict(self, state_dict: Dict[str, Tensor] = None, prefix: str = '') -> Dict[str, Tensor]:
        if state_dict is None:
            state_dict = {}
        if len(prefix) > 0 and not prefix.endswith('.'):
            prefix += '.'
        state_dict[f'{prefix}warmup'] = torch.tensor(self.warmup, device='cpu')
        state_dict[f'{prefix}total_count'] = torch.tensor(self.total_count, device='cpu')
        return super().state_dict(state_dict, prefix)

    def update(self, val, n=1):
        if self.total_count >= self.warmup:
            super().update(val, n=n)
        elif self.total_count + n > self.warmup:
            super().update(val, n=self.total_count + n - self.warmup)
        self.total_count += n

class RunMetrics:
    def __init__(
        self,
        max_acc1: Optional[ValueEpoch] = None,
        min_loss: Optional[ValueEpoch] = None,
        pruning_max_acc1: Optional[Dict[str, ValueEpoch]] = None,
        pruning_min_loss: Optional[Dict[str, ValueEpoch]] = None,
        pruning_min_acc1_wait: int = 0,
        train_epoch_time: Optional[AverageMeter] = None,
        val_epoch_time: Optional[AverageMeter] = None
    ):
        self.max_acc1 = max_acc1 if max_acc1 is not None else ValueEpoch(val=-1, epoch=-1)
        self.min_loss = min_loss if min_loss is not None else ValueEpoch(val=np.inf, epoch=-1)
        self.pruning_max_acc1 = pruning_max_acc1 if pruning_max_acc1 is not None else {}
        self.pruning_min_loss = pruning_min_loss if pruning_min_loss is not None else {}
        self.pruning_min_acc1_wait = pruning_min_acc1_wait
        self.train_epoch_time = train_epoch_time if train_epoch_time is not None else AverageMeter()
        self.val_epoch_time = val_epoch_time if val_epoch_time is not None else AverageMeter()
        self.val_acc1 = 0.0
        self.val_acc5 = 0.0
        self.val_loss = np.inf

class BatchMetrics:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.batch_time: AverageMeter = AverageMeter()
        self.batch_time_warmup: AverageMeterWarmup = AverageMeterWarmup(warmup=20)
        self.imgs_per_sec: AverageMeter = AverageMeter()

    def update(self, batch_size: int, batch_time_sec: float) -> None:
        self.batch_time.update(batch_time_sec)
        self.batch_time_warmup.update(batch_time_sec)

        world_batch_size = batch_size * self.world_size
        self.imgs_per_sec.update(world_batch_size / batch_time_sec)

    def reset(self):
        # Don't reset batch_time_warmup, don't want to warm it up again and only used for estimating ETA
        self.batch_time.reset()
        self.imgs_per_sec.reset()
