import torch

from spconv.layer import SpConv2d
from test.util.build_model import build_model, ModelType, load_checkpoint
from test.util.patcher import patch_conv2d


def main():
    ckpt_path = '/data/sparse-dnn/runs/imnet100/resnet50/finetune/resnet50_imnet100_unstr/ckpt_min_loss_sparsity87.pth'
    rn50 = build_model(100, ModelType.RESNET50)
    epoch, metrics, sparsity, target_sparsity = load_checkpoint(ckpt_path, rn50, None, None)
    print(f'Loaded {ModelType.RESNET50} [sparsity={target_sparsity}, acc@1={metrics.max_acc1.val}]')
    replaced = patch_conv2d(rn50, SpConv2d.from_dense)
    print('Patched Conv2d layers')
    print(f'Replaced modules:\n{replaced}')
    # print(f'New model:\n{rn50}')

    rn50.cuda()
    input = torch.rand(32, 3, 224, 224).cuda()
    out = rn50(input)
    print(out.shape)


if __name__ == '__main__':
    main()
