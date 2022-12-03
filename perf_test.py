print('inside file')
from tkinter import N
import torch

from spconv.format import ELLR
from spconv.functional import sp_conv2d

'''
(1): Bottleneck(
    (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act1): ReLU(inplace=True)
    (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (drop_block): Identity()
    (act2): ReLU(inplace=True)
    (aa): Identity()
    (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act3): ReLU(inplace=True)
)
'''

SKIP_ITERS = 1
TEST_ITERS = 5

def main():
    print('in main')
    n = 32
    c_i = 128
    h = 128
    w = 128
    c_o = 128
    g = 1
    r = 3
    s = 3
    padding = (1, 1)
    # n = 4
    # c_i = 4
    # h = 8
    # w = 8
    # c_o = 2
    # g = 1
    # r = 3
    # s = 3

    torch.manual_seed(2020)

    print('Allocating tensors...')
    inp = torch.rand(n, c_i, h, w).cuda()
    weight = torch.rand(c_o, int(c_i/g), r, s).cuda()
    mask = (torch.rand(c_o, int(c_i/g), r, s) > 0.5).cuda()
    pruned_weight = (weight * mask).cuda()
    pruned_weight_ellr = ELLR.from_dense(pruned_weight.view(c_o, -1), pruned_weight.shape)
    bias = torch.rand(c_o).cuda()

    runtimes_ms = []

    for i in range(SKIP_ITERS + TEST_ITERS):
        print(f'Running iteration {i}')
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        if i >= SKIP_ITERS:
            start_ev.record()

        sp_out = sp_conv2d(
            inp,
            pruned_weight_ellr,
            bias=bias,
            padding=padding
        )
        torch.cuda.synchronize()

        if i >= SKIP_ITERS:
            end_ev.record()
            start_ev.synchronize()
            end_ev.synchronize()
            runtimes_ms.append(start_ev.elapsed_time(end_ev))

    mean_runtime_ms = sum(runtimes_ms) / TEST_ITERS
    print(f'Average runtime: {mean_runtime_ms} ms')
    print(runtimes_ms)

if __name__ == '__main__':
    main()
