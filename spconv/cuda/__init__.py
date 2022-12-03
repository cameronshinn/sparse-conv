import os

import torch
from torch.utils import cpp_extension

src_dir = os.path.dirname(os.path.realpath(__file__))
spconv_src = os.path.join(src_dir, 'spconv.cu')

if torch.cuda.is_available():
    spconv = cpp_extension.load(
        name='spconv',
        sources=[spconv_src],
        with_cuda=True,
        extra_cflags=['-std=c++17'],
        extra_cuda_cflags=['-std=c++17'],#, '-g', '-G', '-O0'],  # NOTE: Make sure these debug flags are commented out for benchmarking
        verbose=True
    )
else:
    raise ImportError('Unable to determine GPU compute capability (no CUDA device found)')

from spconv import spconv2d, spconv2d_v2, spconv2d_v3, spconv2d_v4
