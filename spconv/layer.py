from typing import Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import sp_conv2d

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

        values, col_idx, row_nnz = self._weights_to_ellr()
        self.weight_values = nn.parameter.Parameter(values)
        self.weight_col_idx = nn.parameter.Parameter(col_idx, requires_grad=False)
        self.weight_row_nnz = nn.parameter.Parameter(row_nnz, requires_grad=False)

    def _weights_to_ellr(self):
        out_features = int(self.out_channels / self.groups)
        dn_weight_2d = self.weight.view(out_features, -1)

        row_nnz = torch.count_nonzero(dn_weight_2d, axis=1)
        max_row_nnz = row_nnz.max()

        col_idx = [torch.nonzero(row).view(-1) for row in dn_weight_2d]
        values = [torch.gather(row, 0, idxs) for row, idxs in zip(dn_weight_2d, col_idx)]

        pad_stack = lambda x, z : torch.stack([F.pad(row, (0, max_row_nnz - nnz)) for row, nnz in zip(x, z)])
        col_idx = pad_stack(col_idx, row_nnz)
        values = pad_stack(values, row_nnz)

        return values, col_idx, row_nnz

    # def _weights_to_ellr(self, weight: Optional[nn.Module] = None):
    #     w = self.weight if weight is None else weight
    #     out_features = int(self.out_channels / self.groups)  # Parent class checks even divisibility already
    #     return ELLR.from_dense(self.weight.view(out_features, -1), self.weight.shape)

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

        values, col_idx, row_nnz = sp._weights_to_ellr()
        sp.weight_values = nn.parameter.Parameter(values)
        sp.weight_col_idx = nn.parameter.Parameter(col_idx, requires_grad=False)
        sp.weight_row_nnz = nn.parameter.Parameter(row_nnz, requires_grad=False)
        return sp

    def forward(self, x: torch.Tensor):
        return sp_conv2d(
            x,
            self.weight_values,
            self.weight_col_idx,
            self.weight_row_nnz,
            self.weight.shape,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
