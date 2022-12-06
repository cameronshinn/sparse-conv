from typing import Optional, Union

import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from .format import ELLR
from . import cuda

class SpConv2d(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        input: torch.Tensor,
        weight: ELLR,
        bias: Optional[torch.Tensor],
        stride: Union[int, tuple],
        padding: Union[str, int, tuple],
        dilation: Union[int, tuple],
        groups: int
    ) -> torch.Tensor:
        if groups != 1:
            raise NotImplementedError('Grouped convolution not implemented')

        if bias is None:
            bias = torch.zeros(weight.orig_size[0])

        if isinstance(stride, int):
            stride = (stride, stride)

        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation)

        if padding == 'valid':
            pad = 0
        elif padding == 'same':
            if stride != 1 and stride != (1, 1):
                raise ValueError('"valid" padding mode requires a stride of 1')

            h = weight.orig_size[2]
            w = weight.orig_size[3]
            h_d = (h - 1) * dilation[0] + 1
            w_d = (w - 1) * dilation[1] + 1
            pad = (int(h_d / 2), int((h_d - 1) / 2), int(w_d / 2), int((w_d - 1) / 2))
        elif isinstance(padding, int):
            pad = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) != 2:
                raise ValueError('Padding tuple can only have 2 dimensions')

            pad = (padding[0], padding[0], padding[1], padding[1])
        else:
            raise ValueError('Padding argument must be a string ({"valid", "same"}), int or tuple')

        padded_input = torch.nn.functional.pad(
            input,
            pad
        )

        output = cuda.spconv2d_v4(  # NOTE: Change this to test different kernels
            padded_input,
            weight.values,
            weight.col_idx,
            weight.row_nnz,
            weight.orig_size[0],
            weight.orig_size[1],
            weight.orig_size[2],
            weight.orig_size[3],
            bias,
            stride[0],
            stride[1],
            dilation[0],
            dilation[1]
        )
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: torch.Tensor
    ) -> torch.Tensor:
        pass


def sp_conv2d(
    input: torch.Tensor,
    weight: ELLR,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, tuple] = 1,
    padding: Union[str, int, tuple] = 0,
    dilation: Union[int, tuple] = 1,
    groups: int = 1
):
    return SpConv2d.apply(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups
    )
