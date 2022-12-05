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

def main():
    n = 16
    c_i = 64
    h = 32
    w = 32
    c_o = 64
    g = 1
    r = 3
    s = 3
    padding = (1, 1)
    # n = 2
    # c_i = 3
    # h = 6
    # w = 6
    # c_o = 2
    # g = 1
    # r = 3
    # s = 3

    print_tensors = True

    torch.manual_seed(2020)

    inp = torch.rand(n, c_i, h, w).cuda()
    weight = torch.rand(c_o, int(c_i/g), r, s).cuda()
    mask = (torch.rand(c_o, int(c_i/g), r, s) > 0.8).cuda()
    pruned_weight = (weight * mask).cuda()
    pruned_weight_ellr = ELLR.from_dense(pruned_weight.view(c_o, -1), pruned_weight.shape)
    bias = torch.rand(c_o).cuda()

    if print_tensors:
        print(f'input:\n{inp}\n')
        print(f'weight:\n{pruned_weight}\n')
        print(f'bias:\n{bias}\n')

    sp_out = sp_conv2d(
        inp,
        pruned_weight_ellr,
        bias=bias,
        padding=padding
    )
    out = torch.nn.functional.conv2d(
        inp,
        pruned_weight,
        bias=bias,
        padding=padding
    )

    if print_tensors:
        print(sp_out)
        print(out)

    if sp_out.allclose(out):
        print('PASS')
    else:
        print('FAIL')


if __name__ == '__main__':
    main()
