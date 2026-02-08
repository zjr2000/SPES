import torch
import torch.nn as nn
from typing import Any

import stk
import stk.backend.triton_kernels
import stk.ops
import torch
from packaging import version

from megablocks import grouped_gemm_util as gg
from megablocks.layers import common, gelu, mpu
from megablocks.layers.activation_fn import act_fn
from megablocks.layers.arguments import DEFAULT_ACTIVATION_FN, Arguments, InitFn
from megablocks.layers.mlp import (
    create_dmoe_expert_weights,
    resolve_dtensor,
)



class CustomSparseGLU(torch.nn.Module):
    def __init__(self, args: Arguments):
        super().__init__()
        self.args = args

        # 本 rank local 的 expert 数+每个 expert 的中间隐藏维
        self.num_local_experts = mpu.experts_per_rank(args)
        self.hidden_size = args.hidden_size

        # ------------------------------------------------------------------ #
        # 1. 创建 ParameterList
        # ------------------------------------------------------------------ #
        self.expert_w1: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        args.ffn_hidden_size,
                        self.hidden_size,
                        device=args.device,
                        dtype=common.dtype(args),
                    )
                )
                for _ in range(self.num_local_experts)
            ]
        )
        self.expert_v1: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        args.ffn_hidden_size,
                        self.hidden_size,
                        device=args.device,
                        dtype=common.dtype(args),
                    )
                )
                for _ in range(self.num_local_experts)
            ]
        )
        self.expert_w2: nn.ParameterList = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        args.ffn_hidden_size,
                        self.hidden_size,
                        device=args.device,
                        dtype=common.dtype(args),
                    )
                )
                for _ in range(self.num_local_experts)
            ]
        )

        # ------------------------------------------------------------------ #
        # 2. 初始化
        # ------------------------------------------------------------------ #
        # 依旧借助 create_dmoe_expert_weights，然后按行切块复制到各 expert
        with torch.no_grad():
            # w1
            full_w1 = create_dmoe_expert_weights(
                args,
                args.moe_num_experts,
                args.ffn_hidden_size,
                args.hidden_size,
                args.init_method,
            )
            # v1
            full_v1 = create_dmoe_expert_weights(
                args,
                args.moe_num_experts,
                args.ffn_hidden_size,
                args.hidden_size,
                args.init_method,
            )
            # w2
            full_w2 = create_dmoe_expert_weights(
                args,
                args.moe_num_experts,
                args.ffn_hidden_size,
                args.hidden_size,
                args.output_layer_init_method,
            )

            # 把属于本 rank 的 slice 拆给每个 expert
            rows_per_expert = args.ffn_hidden_size
            for idx in range(self.num_local_experts):
                r0 = idx * rows_per_expert
                r1 = r0 + rows_per_expert
                self.expert_w1[idx].copy_(full_w1[r0:r1])
                self.expert_v1[idx].copy_(full_v1[r0:r1])
                self.expert_w2[idx].copy_(full_w2[r0:r1])

        # ------------------------------------------------------------------ #
        # 3. expert parallelism 属性和渐变缩放
        # ------------------------------------------------------------------ #
        self._should_set_parallelism_attribute = args.moe_expert_model_parallelism
        for plist in (self.expert_w1, self.expert_v1, self.expert_w2):
            for p in plist:
                mpu.set_expert_model_parallel_attributes(
                    p,
                    self._should_set_parallelism_attribute,
                )

        self.gradient_scale = None
        if args.moe_expert_model_parallelism:
            self.gradient_scale = 1.0 / mpu.get_expert_parallel_world_size(args)

    # helper: 梯度缩放
    def _scale_grad(self, w: torch.Tensor) -> torch.Tensor:
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    # ---------------------------------------------------------------------- #
    # 4. forward
    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, topo: Any):
        if self.args.memory_optimized_mlp:
            raise NotImplementedError(
                'Memory optimized implementation not yet supported with GLU with sparse kernels.',
            )

        # 先把各 expert 的参数取出、做 grad-scale、resolve_dtensor
        w1_parts: List[torch.Tensor] = [
            self._scale_grad(resolve_dtensor(p)) for p in self.expert_w1
        ]
        v1_parts: List[torch.Tensor] = [
            self._scale_grad(resolve_dtensor(p)) for p in self.expert_v1
        ]
        w2_parts: List[torch.Tensor] = [
            self._scale_grad(resolve_dtensor(p)) for p in self.expert_w2
        ]

        # 在 forward 时临时拼成一个大矩阵；随后立即参与计算
        w1 = torch.cat(w1_parts, dim=0)  # (E*F, H)
        v1 = torch.cat(v1_parts, dim=0)
        w2 = torch.cat(w2_parts, dim=0)

        # ---- GLU ----
        x1 = stk.ops.sdd(x, w1.t().contiguous(), topo)
        x2 = stk.ops.sdd(x, v1.t().contiguous(), topo)

        act_out = act_fn(x1, self.args.activation_fn)
        x1 = stk.ops.mul(act_out, x2)

        out = stk.ops.dsd(x1, w2.contiguous())
        return out
