import torch
import torch.nn.functional as F

class ELLR:
    def __init__(self, values: torch.Tensor, col_idx: torch.Tensor, row_nnz: torch.Tensor, orig_size: torch.Size) -> None:
        self.values = values
        self.col_idx = col_idx
        self.row_nnz = row_nnz
        self.orig_size = orig_size

    @classmethod
    def from_dense(cls, dense: torch.Tensor, orig_size: torch.Size):
        if dense.dim() != 2:
            raise ValueError('Expected 2D tensor input')

        row_nnz = torch.count_nonzero(dense, axis=1)
        max_row_nnz = row_nnz.max()

        col_idx = [torch.nonzero(row).view(-1) for row in dense]
        values = [torch.gather(row, 0, idxs) for row, idxs in zip(dense, col_idx)]

        pad_stack = lambda x, z : torch.stack([F.pad(row, (0, max_row_nnz - nnz)) for row, nnz in zip(x, z)])
        col_idx = pad_stack(col_idx, row_nnz)
        values = pad_stack(values, row_nnz)

        return cls(values, col_idx, row_nnz, orig_size)
