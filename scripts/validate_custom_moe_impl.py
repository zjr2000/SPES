import os
os.environ['PYTHONHASHSEED'] = '42'
import torch
import stk
from torch import nn
import numpy as np
import megablocks.ops as ops
from typing import Any, Callable, Union
from stk import Matrix
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def act_fn(
    x: Matrix,
    function: Callable,
    return_grad_fn: bool = False,
    **kwargs,
) -> Union[tuple[Matrix, Any] | Matrix]:
    assert isinstance(x, Matrix)
    with torch.set_grad_enabled(torch.is_grad_enabled() or return_grad_fn):
        if return_grad_fn:
            x.data.requires_grad = True
        out = function(x.data, **kwargs)
        y = Matrix(
            x.size(),
            out,
            x.row_indices,
            x.column_indices,
            x.offsets,
            x.column_indices_t,
            x.offsets_t,
            x.block_offsets_t,
        )
        if return_grad_fn:
            return y, out.backward
        return y

def sparse_transpose(size, row_indices, column_indices, offsets, blocking=128):
    block_columns = size[1] // blocking
    max_column_index = ((128 * 4) // blocking)
    transpose_sort_end_bit = max(
        int(np.ceil(np.log2(max_column_index))),
        1,
    )
    _, gather_indices = ops.sort(
        column_indices.int(),
        transpose_sort_end_bit,
    )
    column_indices_t = row_indices.gather(0, gather_indices.long())
    block_offsets_t = gather_indices.int()
    zero = torch.zeros((1,), dtype=torch.int32, device=row_indices.device)
    nnz_per_column = ops.histogram(column_indices, block_columns)
    nnz_per_column = ops.inclusive_cumsum(nnz_per_column, 0)
    if nnz_per_column.dim() == 0:
        nnz_per_column = nnz_per_column.unsqueeze(0)
    offsets_t = torch.cat([zero, nnz_per_column])
    return column_indices_t, offsets_t, block_offsets_t

class SparseMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

class OriginalSparseGLU(SparseMLP):
    def __init__(self, args):
        super().__init__(args)
        self.w1 = nn.Parameter(torch.ones(args.num_experts*args.ffn_dim, args.hidden_size))
        self.v1 = nn.Parameter(torch.ones(args.num_experts*args.ffn_dim, args.hidden_size)) 
        self.w2 = nn.Parameter(torch.ones(args.num_experts*args.ffn_dim, args.hidden_size))

    def forward(self, x, topo):
        w1, v1, w2 = self.w1, self.v1, self.w2
        x1 = stk.ops.sdd(x, w1.t(), topo)
        x2 = stk.ops.sdd(x, v1.t(), topo)
        activation_fn_out = act_fn(x1, torch.nn.functional.gelu)
        x1 = stk.ops.mul(activation_fn_out, x2)
        return stk.ops.dsd(x1, w2)

class ExpertWiseSparseGLU(SparseMLP):
    def __init__(self, args):
        super().__init__(args)
        self.num_experts = args.num_experts
        self.expert_w1 = nn.ParameterList([
            nn.Parameter(torch.ones(args.ffn_dim, args.hidden_size)) for _ in range(self.num_experts)
        ])
        self.expert_v1 = nn.ParameterList([
            nn.Parameter(torch.ones(args.ffn_dim, args.hidden_size)) for _ in range(self.num_experts)
        ])
        self.expert_w2 = nn.ParameterList([
            nn.Parameter(torch.ones(args.ffn_dim, args.hidden_size)) for _ in range(self.num_experts)
        ])

    def forward(self, x, topo):
        w1 = torch.cat([e_w1 for e_w1 in self.expert_w1], dim=0)
        v1 = torch.cat([e_v1 for e_v1 in self.expert_v1], dim=0)
        w2 = torch.cat([e_w2 for e_w2 in self.expert_w2], dim=0)

        # w2 = self.w2
        x1 = stk.ops.sdd(x, w1.t().contiguous(), topo)
        x2 = stk.ops.sdd(x, v1.t().contiguous(), topo)
        activation_fn_out = act_fn(x1, torch.nn.functional.gelu)
        x1 = stk.ops.mul(activation_fn_out, x2)
        return stk.ops.dsd(x1, w2.contiguous())

def get_dense_topo(x, args, blocking=128):
    padded_tokens = x.shape[0]
    num_experts = args.num_experts
    ffn_hidden_size = args.ffn_dim
    device = x.device
    dtype = x.dtype
    assert padded_tokens % blocking == 0, "batch 必须能被 blocking 整除"
    block_rows = padded_tokens // blocking
    blocks_per_row = (ffn_hidden_size * num_experts) // blocking
    offsets = torch.arange(
        0,
        block_rows * blocks_per_row + 1,
        blocks_per_row,
        dtype=torch.int32,
        device=device,
    )
    column_indices = torch.arange(blocks_per_row, device=device, dtype=torch.int32).repeat(block_rows)
    nnz = column_indices.numel()
    data = torch.ones(
        nnz,
        blocking,
        blocking,
        dtype=dtype,
        device='meta',
    )
    shape = (padded_tokens, ffn_hidden_size * num_experts)
    row_indices = stk.ops.row_indices(shape, data, offsets, column_indices)
    column_indices_t, offsets_t, block_offsets_t = sparse_transpose(
        shape,
        row_indices,
        column_indices,
        offsets,
    )
    topo = stk.Matrix(
        shape,
        data,
        row_indices,
        column_indices,
        offsets,
        column_indices_t,
        offsets_t,
        block_offsets_t,
    )
    return topo

if __name__ == "__main__":
    class Args:
        num_experts = 4
        ffn_dim = 128
        hidden_size = 256
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. OriginalSparseGLU 独立初始化、前向、反向
    args1 = Args()
    model_original = OriginalSparseGLU(args1).to(args1.device)
    x1 = torch.ones(128, args1.hidden_size).to(args1.device)
    print(x1.sum())
    topo1 = get_dense_topo(x1, args1, blocking=128)
    print("OriginalSparseGLU forward ...")
    out1 = model_original(x1, topo1)
    print(out1)
    loss1 = out1.sum()
    loss1.backward()
    print("OriginalSparseGLU loss:", loss1.item())

    # 2. ExpertWiseSparseGLU 独立初始化、前向、反向
    args2 = Args()
    model_expert = ExpertWiseSparseGLU(args2).to(args2.device)
    x2 = torch.ones(128, args2.hidden_size).to(args2.device)
    # print(x2.sum())
    topo2 = get_dense_topo(x2, args2, blocking=128)
    print("ExpertWiseSparseGLU forward ...")
    out2 = model_expert(x2, topo2)
    print(out2)
    loss2 = out2.sum()
    loss2.backward()
    print("ExpertWiseSparseGLU loss:", loss2.item())
