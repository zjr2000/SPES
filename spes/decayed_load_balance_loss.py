from megablocks.layers.moe import get_load_balancing_loss
from megablocks.layers.arguments import Arguments 
import torch

_TRAINABLE_EXPERT_INDICES = []
_TRAINABLE_EXPERT_LOSS_FACTOR = 1.0

def set_trainable_expert_indices(indices):
    global _TRAINABLE_EXPERT_INDICES
    # 避免重复添加
    _TRAINABLE_EXPERT_INDICES = list(set(indices))

def get_trainable_expert_indices():
    global _TRAINABLE_EXPERT_INDICES
    return _TRAINABLE_EXPERT_INDICES

def set_current_trainable_expert_loss_factor(step, total_step):
    """
    初始值0.7, 训练前20% linearly衰减到1.0, 每500步更新一次。
    保证逐步递增到1.0, 后续固定
    """
    global _TRAINABLE_EXPERT_LOSS_FACTOR
    start = 0.7
    end = 1.0
    decay_ratio = 0.2
    decay_steps = int(total_step * decay_ratio)
    if step < decay_steps:
        # Linear增大
        factor = start + (end - start) * (step / max(1, decay_steps))
    else:
        factor = end
    # 每500步动更新一次（其实只需调用方外面控制）
    _TRAINABLE_EXPERT_LOSS_FACTOR = float(factor)

def get_current_trainable_expert_loss_factor():
    global _TRAINABLE_EXPERT_LOSS_FACTOR
    return _TRAINABLE_EXPERT_LOSS_FACTOR

def batched_load_balancing_loss(args: Arguments):
    if args.moe_loss_weight == 0:
        return 0.0

    tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
    num_layers_per_pipeline_stage = args.num_layers // args.pipeline_model_parallel_size
    if args.num_layers_per_virtual_pipeline_stage is not None:
        num_layers_per_pipeline_stage = args.num_layers_per_virtual_pipeline_stage

    if len(tokens_per_expert) != num_layers_per_pipeline_stage:
        raise ValueError(
            f'Expected {num_layers_per_pipeline_stage} token_per_experts '
            f'but found {len(tokens_per_expert)}.\nnum_layers = '
            f'{args.num_layers}\npipeline_model_parallel_size = '
            f'{args.pipeline_model_parallel_size}\n'
            'num_layers_per_virtual_pipeline_stage'
            f' = {args.num_layers_per_virtual_pipeline_stage}',
        )
    if len(expert_scores) != num_layers_per_pipeline_stage:
        raise ValueError(
            f'Expected {num_layers_per_pipeline_stage} expert_scores '
            f'but found {len(tokens_per_expert)}.\nnum_layers = '
            f'{args.num_layers}\npipeline_model_parallel_size = '
            f'{args.pipeline_model_parallel_size}\n'
            'num_layers_per_virtual_pipeline_stage'
            f' = {args.num_layers_per_virtual_pipeline_stage}',
        )

    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all((x.ndim == 1 and x.numel() == args.moe_num_experts for x in tokens_per_expert))

    tokens = expert_scores[0].shape[0]
    assert all(((x.ndim == 2 and x.shape[1] == args.moe_num_experts and x.shape[0] == tokens) for x in expert_scores))

    # Concatenate layer contributions:
    expert_scores = torch.cat(expert_scores, dim=1)
    if args.moe_lbl_in_fp32:
        expert_scores = expert_scores.float()
    # Reduce tokens dim:
    if tokens != 0:
        expert_scores = expert_scores.mean(dim=0)
    else:
        expert_scores = expert_scores.sum(dim=0)
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

    expected_values = num_layers_per_pipeline_stage * args.moe_num_experts
    assert tokens_per_expert.numel() == expected_values
    assert expert_scores.numel() == expected_values

    # ----------- 这里加load balance调整 -------------
    # 获取trainable expert index和factor
    trainable_indices = get_trainable_expert_indices()
    factor = get_current_trainable_expert_loss_factor()  # e.g. 0.7~1.0
    weights = torch.ones_like(tokens_per_expert)
    if trainable_indices:
        # 全局expert index，应每层都平铺排列（如4 expert 2层: [0,1,2,3, 0,1,2,3]）
        for idx in trainable_indices:
            for layer in range(num_layers_per_pipeline_stage):
                global_idx = layer * args.moe_num_experts + idx
                weights[global_idx] = factor

    tokens_per_expert = tokens_per_expert * weights

    # -----------------------------------------------
    # Calculate the total scale as usual
    scale_numerator = (args.moe_num_experts * args.moe_loss_weight)
    scale_denominator = (args.num_layers * tokens * args.moe_top_k)
    scale = scale_numerator / scale_denominator
    return scale * torch.dot(tokens_per_expert, expert_scores)
