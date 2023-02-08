import torch.nn as nn
from typing import Callable, Dict

def patch_conv2d(model: nn.Module, patch_func: Callable[[nn.Module], nn.Module]) -> None:
    patch_modules: Dict[str, nn.Module] = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == module.out_channels:  # FIXME: Only checking channels cause only equal channels is supported thus far
            patch_module = patch_func(module)
            patch_modules[name] = patch_module

    removed_modules: Dict[str, nn.Module] = {}

    for name, patch_module in patch_modules.items():
        removed_modules[name] = model.get_submodule(name)

        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)  # some.module.name -> some.module, name
        else:
            parent_name = ''
            child_name = name

        parent_module = model.get_submodule(parent_name)
        setattr(parent_module, child_name, patch_module)

    return removed_modules
