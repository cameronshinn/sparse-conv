# Sparse Convolution Layer

Drop-in replacement PyTorch layer for sparse convolution. Calls an autograd-extended sparse convolution function that uses a CUDA backend. Backwards method is not supported currently.

## Example

```
from spconv.layer import SpConv2d

layer = SpConv2d.from_dense(my_conv2d_layer)

out = layer(in)
```

A more detailed sample that uses the functional interface can be found in `correct_test.py`.
