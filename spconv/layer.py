from typing import Tuple, TypeVar, Union

import torch
import torch.nn as nn

from format import ELLR
from functional import sp_conv2d

T = TypeVar('T')
_size_2_t = Union[T, Tuple[T, T]]

class SpConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )
        if in_channels != out_channels:
            raise NotImplementedError('Only `in_channels == out_channels` is supported')

        # FIXME: Need to reshape weight to be 2d and save original dimensions
        self.weight = ELLR.from_dense(self.weight)  # Initialize to empty ELLR matrix

    @classmethod
    def from_dense(cls, dn: nn.Conv2d):
        sp = cls(
            dn.in_channels,
            dn.out_channels,
            dn.kernel_size,
            stride=dn.stride,
            padding=dn.padding,
            dilation=dn.dilation,
            groups=dn.groups,
            bias=(dn.bias is not None),
            padding_mode=dn.padding_mode
        )

        sp.weight = ELLR.from_dense(dn.weight)
        return sp

    def forward(self, x: torch.Tensor):
        pass

    def backward(self):
        pass
