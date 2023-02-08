import numpy as np
from timm.models.convnext import convnext_tiny, convnext_base
from timm.models.efficientnet import efficientnet_b4, efficientnet_b7
from timm.models.resnet import resnet50, resnet152
from timm.scheduler.scheduler import Scheduler
import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from typing import Any, Dict, Optional, Tuple

from .ext_enum import ExtendedEnum
from .metrics import AverageMeter, ValueEpoch, RunMetrics

class ModelType(ExtendedEnum):
    RESNET50 = 'resnet50'
    RESNET152 = 'resnet152'
    EFFICIENTNET_B4 = 'efficientnet_b4'
    EFFICIENTNET_B7 = 'efficientnet_b7'
    CONVNEXT_TINY = 'convnext_tiny'
    CONVNEXT_BASE = 'convnext_base'

def build_model(
    num_classes: int,
    model_type: ModelType,
    pretrained: bool = False,
) -> nn.Module:
    if model_type == ModelType.CONVNEXT_TINY:
        model = convnext_tiny(pretrained=pretrained, num_classes=num_classes)
    elif model_type == ModelType.CONVNEXT_BASE:
        model = convnext_base(pretrained=pretrained, num_classes=num_classes)
    elif model_type == ModelType.EFFICIENTNET_B4:
        model = efficientnet_b4(pretrained=pretrained, drop_rate=0.4, drop_path_rate=0.2, num_classes=num_classes)
    elif model_type == ModelType.EFFICIENTNET_B7:
        model = efficientnet_b7(pretrained=pretrained, drop_rate=0.5, drop_path_rate=0.2, num_classes=num_classes)
    elif model_type == ModelType.RESNET50:
        model = resnet50(pretrained=pretrained, num_classes=num_classes)
    elif model_type == ModelType.RESNET152:
        model = resnet152(pretrained=pretrained, num_classes=num_classes)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented. Choices: {ModelType.list()}')

    return model


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    scaler: Optional[GradScaler],
    optimizer: Optional[Optimizer]
) -> Tuple[int, RunMetrics, float, float]:

    checkpoint: Dict[str, Any]
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    max_acc1 = ValueEpoch(val=0, epoch=-1)
    min_loss = ValueEpoch(val=np.inf, epoch=-1)
    pruning_max_acc1: Dict[str, ValueEpoch] = {}
    pruning_min_loss: Dict[str, ValueEpoch] = {}
    epoch = -1
    sparsity = 0.0
    target_sparsity = 0.0
    pruning_min_acc1_wait = 0
    train_epoch_time = AverageMeter()
    val_epoch_time = AverageMeter()

    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scaler' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    if 'max_acc1' in checkpoint:
        max_acc1.load_state_dict(checkpoint['max_acc1'])
    if 'min_loss' in checkpoint:
        min_loss.load_state_dict(checkpoint['min_loss'])

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
    if 'train_epoch_time' in checkpoint:
        train_epoch_time.load_state_dict(checkpoint['train_epoch_time'])
    if 'val_epoch_time' in checkpoint:
        val_epoch_time.load_state_dict(checkpoint['val_epoch_time'])
    if 'sparsity' in checkpoint:
        sparsity = checkpoint['sparsity']
    if 'target_sparsity' in checkpoint:
        target_sparsity = checkpoint['target_sparsity']
    if 'pruning_max_acc1' in checkpoint:
        pruning_max_acc1 = checkpoint['pruning_max_acc1']
    if 'pruning_min_loss' in checkpoint:
        pruning_min_loss = checkpoint['pruning_min_loss']
    if 'pruning_min_acc1_wait' in checkpoint:
        pruning_min_acc1_wait = checkpoint['pruning_min_acc1_wait']

    metrics = RunMetrics(
        max_acc1=max_acc1,
        min_loss=min_loss,
        pruning_max_acc1=pruning_max_acc1,
        pruning_min_loss=pruning_min_loss,
        pruning_min_acc1_wait=pruning_min_acc1_wait,
        train_epoch_time=train_epoch_time,
        val_epoch_time=val_epoch_time
    )

    del checkpoint
    torch.cuda.empty_cache()

    return epoch, metrics, sparsity, target_sparsity
