#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从本地 Qwen3 dense 模型初始化一个 OLMoE 模型（Random Noise Upcycling + 可定制 Router 初始化 + 参数量统计）。

依赖：
    pip install "transformers>=4.46.0" torch

使用示例：
    python qwen3_to_olmoe_local.py \
        --qwen3_path ./Qwen3-8B \
        --save_dir ./OLMoE-from-Qwen3 \
        --num_local_experts 8 \
        --num_experts_per_tok 2 \
        --noise_std 0.02 \
        --noise_fraction 0.5 \
        --router_init_strategy qwen_mlp_based \
        --device cuda \
        --dtype float16
"""

import argparse
import math
from typing import Tuple

import torch
from transformers import (
    Qwen3ForCausalLM,
    Qwen3Config,
    OlmoeForCausalLM,
    OlmoeConfig,
)

# ---------------------- 参数统计工具 ---------------------- #

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n} ({n/1e9:.3f} B)"
    if n >= 1_000_000:
        return f"{n} ({n/1e6:.3f} M)"
    if n >= 1_000:
        return f"{n} ({n/1e3:.3f} K)"
    return str(n)


def estimate_olmoe_activation_params(model: OlmoeForCausalLM) -> int:
    """
    估算 OLMoE 的每 token 平均激活参数量：
      dense 参数全部激活；
      MoE 参数按 (topk / num_local_experts) 折算。
    """
    config: OlmoeConfig = model.config
    num_experts = config.num_local_experts
    topk = config.num_experts_per_tok

    dense_params = 0
    moe_params = 0

    for name, p in model.named_parameters():
        if "mlp.experts." in name:
            moe_params += p.numel()
        else:
            dense_params += p.numel()

    moe_activation = moe_params * (topk / num_experts)
    activation_params = dense_params + moe_activation
    return int(activation_params)


def estimate_qwen_activation_params(model: Qwen3ForCausalLM) -> int:
    return count_parameters(model)

# ---------------------- 构造 OlmoeConfig ---------------------- #

def build_olmoe_config_from_qwen3(
    q_config: Qwen3Config,
    num_local_experts: int = 8,
    num_experts_per_tok: int = 2,
    router_aux_loss_coef: float = 0.001,
) -> OlmoeConfig:
    """
    根据 Qwen3Config 自动构造一个 OlmoeConfig，使两者在基本结构上兼容。
    """
    o_config = OlmoeConfig(
        vocab_size=q_config.vocab_size,
        hidden_size=q_config.hidden_size,
        intermediate_size=q_config.intermediate_size,
        num_hidden_layers=q_config.num_hidden_layers,
        num_attention_heads=q_config.num_attention_heads,
        num_key_value_heads=q_config.num_key_value_heads,
        max_position_embeddings=q_config.max_position_embeddings,
        rms_norm_eps=getattr(q_config, "rms_norm_eps", 1e-5),
        hidden_act=q_config.hidden_act,
        attention_dropout=q_config.attention_dropout,
        pad_token_id=q_config.pad_token_id,
        rope_scaling=q_config.rope_scaling,
        rope_theta=q_config.rope_theta,
        # MoE 相关
        num_local_experts=num_local_experts,
        num_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=True,
        router_aux_loss_coef=router_aux_loss_coef,
        # 其他
        attention_bias=getattr(q_config, "attention_bias", False),
        sliding_window=getattr(q_config, "sliding_window", None),
        clip_qkv=getattr(q_config, "clip_qkv", None),
        output_router_logits=True,
        initializer_range=getattr(q_config, "initializer_range", 0.02),
    )
    return o_config


# ---------------------- Embedding & lm_head 拷贝 ---------------------- #

def copy_embedding_and_lm_head(qwen: Qwen3ForCausalLM, olmoe: OlmoeForCausalLM):
    # 1. token embedding
    assert qwen.model.embed_tokens.weight.shape[1] == olmoe.model.embed_tokens.weight.shape[1], \
        f"hidden_size mismatch: qwen={qwen.model.embed_tokens.weight.shape[1]}, olmoe={olmoe.model.embed_tokens.weight.shape[1]}"
    vocab_min = min(qwen.model.embed_tokens.weight.shape[0], olmoe.model.embed_tokens.weight.shape[0])

    with torch.no_grad():
        olmoe.model.embed_tokens.weight[:vocab_min].copy_(qwen.model.embed_tokens.weight[:vocab_min])

    # 2. lm_head（tied，但这里直接拷一份）
    with torch.no_grad():
        olmoe.lm_head.weight[:vocab_min].copy_(qwen.lm_head.weight[:vocab_min])


# ---------------------- Attention 拷贝 ---------------------- #

def copy_attention_layer(
    q_layer,
    o_layer,
    layer_idx: int,
    q_config: Qwen3Config,
    o_config: OlmoeConfig,
):
    """
    将单层 Qwen3DecoderLayer.self_attn 映射到 OlmoeDecoderLayer.self_attn。
    """
    q_attn = q_layer.self_attn
    o_attn = o_layer.self_attn

    with torch.no_grad():
        # 1. 线性层权重 (q, k, v, o)
        assert q_attn.q_proj.weight.shape == o_attn.q_proj.weight.shape, \
            f"q_proj weight mismatch at layer {layer_idx}"
        assert q_attn.k_proj.weight.shape == o_attn.k_proj.weight.shape, \
            f"k_proj weight mismatch at layer {layer_idx}"
        assert q_attn.v_proj.weight.shape == o_attn.v_proj.weight.shape, \
            f"v_proj weight mismatch at layer {layer_idx}"
        assert q_attn.o_proj.weight.shape == o_attn.o_proj.weight.shape, \
            f"o_proj weight mismatch at layer {layer_idx}"

        o_attn.q_proj.weight.copy_(q_attn.q_proj.weight)
        o_attn.k_proj.weight.copy_(q_attn.k_proj.weight)
        o_attn.v_proj.weight.copy_(q_attn.v_proj.weight)
        o_attn.o_proj.weight.copy_(q_attn.o_proj.weight)

        if q_attn.q_proj.bias is not None and o_attn.q_proj.bias is not None:
            o_attn.q_proj.bias.copy_(q_attn.q_proj.bias)
            o_attn.k_proj.bias.copy_(q_attn.k_proj.bias)
            o_attn.v_proj.bias.copy_(q_attn.v_proj.bias)
            o_attn.o_proj.bias.copy_(q_attn.o_proj.bias)

        # 2. q_norm:
        o_attn.q_norm.weight.copy_(q_attn.q_norm.weight)

        # 3. k_norm:
        o_attn.k_norm.weight.copy_(q_attn.k_norm.weight)


# ---------------------- FFN -> MoE experts + Random Noise Upcycling ---------------------- #

def add_random_noise_to_fraction_(tensor: torch.Tensor, fraction: float, std: float):
    """
    在 tensor 中随机选取 fraction 比例的元素，并对这些元素加上 N(0, std) 噪声（in-place）。
    """

    numel = tensor.numel()
    k = int(numel * fraction)

    flat = tensor.view(-1)
    device = flat.device

    idx = torch.randperm(numel, device=device)[:k]
    noise = torch.randn(k, device=device, dtype=flat.dtype) * std
    flat[idx] += noise


def copy_mlp_to_moe_experts(
    q_layer,
    o_layer,
    layer_idx: int,
    q_config: Qwen3Config,
    o_config: OlmoeConfig,
    noise_std: float = 0.02,
    noise_fraction: float = 0.5,
):
    """
    将 Qwen3 的 dense MLP 权重复制到 Olmoe 的所有 experts (OlmoeMLP) 上，
    然后在每个 expert 的 gate_proj / up_proj / down_proj 上注入 Random Noise Upcycling 噪声。
    """
    q_mlp = q_layer.mlp          # Qwen3 MLP
    moe_block = o_layer.mlp      # OlmoeSparseMoeBlock
    experts = moe_block.experts  # ModuleList[OlmoeMLP]

    num_experts = moe_block.num_experts

    # 形状检查：Qwen3 与单个 expert 的 MLP 应该一致
    with torch.no_grad():
        sample_expert = experts[0]

        assert q_mlp.gate_proj.weight.shape == sample_expert.gate_proj.weight.shape, (
            f"[Layer {layer_idx}] gate_proj shape mismatch: "
            f"qwen {q_mlp.gate_proj.weight.shape} vs olmoe {sample_expert.gate_proj.weight.shape}"
        )
        assert q_mlp.up_proj.weight.shape == sample_expert.up_proj.weight.shape, (
            f"[Layer {layer_idx}] up_proj shape mismatch: "
            f"qwen {q_mlp.up_proj.weight.shape} vs olmoe {sample_expert.up_proj.weight.shape}"
        )
        assert q_mlp.down_proj.weight.shape == sample_expert.down_proj.weight.shape, (
            f"[Layer {layer_idx}] down_proj shape mismatch: "
            f"qwen {q_mlp.down_proj.weight.shape} vs olmoe {sample_expert.down_proj.weight.shape}"
        )

        for e in range(num_experts):
            expert = experts[e]

            # 1. 先完全拷贝 dense MLP 权重
            expert.gate_proj.weight.copy_(q_mlp.gate_proj.weight)
            expert.up_proj.weight.copy_(q_mlp.up_proj.weight)
            expert.down_proj.weight.copy_(q_mlp.down_proj.weight)

            # 2. 再对部分权重加 N(0, std) 噪声（Random Noise Upcycling）
            if noise_std is not None and noise_std > 0 and noise_fraction > 0:
                print('add noise')
                add_random_noise_to_fraction_(expert.gate_proj.weight, fraction=noise_fraction, std=noise_std)
                add_random_noise_to_fraction_(expert.up_proj.weight, fraction=noise_fraction, std=noise_std)
                add_random_noise_to_fraction_(expert.down_proj.weight, fraction=noise_fraction, std=noise_std)


# ---------------------- LayerNorm 拷贝 ---------------------- #

def copy_layernorms(q_layer, o_layer, layer_idx: int):
    """
    Qwen3DecoderLayer:
        input_layernorm: Qwen3RMSNorm(hidden_size)
        post_attention_layernorm: Qwen3RMSNorm(hidden_size)

    OlmoeDecoderLayer:
        input_layernorm: OlmoeRMSNorm(hidden_size)
        post_attention_layernorm: OlmoeRMSNorm(hidden_size)
    """
    with torch.no_grad():
        assert q_layer.input_layernorm.weight.shape == o_layer.input_layernorm.weight.shape
        assert q_layer.post_attention_layernorm.weight.shape == o_layer.post_attention_layernorm.weight.shape

        o_layer.input_layernorm.weight.copy_(q_layer.input_layernorm.weight)
        o_layer.post_attention_layernorm.weight.copy_(q_layer.post_attention_layernorm.weight)


def copy_final_norm(qwen: Qwen3ForCausalLM, olmoe: OlmoeForCausalLM):
    q_norm = qwen.model.norm
    o_norm = olmoe.model.norm
    with torch.no_grad():
        assert q_norm.weight.shape == o_norm.weight.shape
        o_norm.weight.copy_(q_norm.weight)


# ---------------------- Router 初始化（可定制） ---------------------- #

def init_router(
    router_module,
    strategy: str = "zeros",
    q_layer=None,
    o_config: OlmoeConfig = None,
    layer_idx: int = 0,
):
    """
    初始化单层的 router 参数。

    注意：在当前 OLMoE 实现中，router 是 OlmoeSparseMoeBlock.gate (nn.Linear)。
    """
    with torch.no_grad():
        w = router_module.weight

        if strategy == "zeros":
            w.zero_()

        elif strategy == "normal":
            torch.nn.init.normal_(w, mean=0.0, std=1.0)

        elif strategy == "scaled_normal":
            torch.nn.init.normal_(w, mean=0.0, std=0.02)

        elif strategy == "qwen_mlp_based":
            if q_layer is None:
                raise ValueError("q_layer is required for 'qwen_mlp_based' router init")
            q_mlp = q_layer.mlp
            down_w = q_mlp.down_proj.weight.data
            std = down_w.std().item()
            if std <= 0:
                std = 0.02
            torch.nn.init.normal_(w, mean=0.0, std=std)

        else:
            raise ValueError(f"Unknown router_init_strategy: {strategy}")

        # 如果 router 有 bias，则一并初始化（目前 gate 没有 bias，这里做兼容）
        if hasattr(router_module, "bias") and router_module.bias is not None:
            if strategy in ["zeros", "qwen_mlp_based"]:
                router_module.bias.zero_()
            else:
                torch.nn.init.zeros_(router_module.bias)


# ---------------------- q/k norm 模块重新初始化（按你的要求：替换成 head_dim 的 RMSNorm） ---------------------- #

def reinit_olmoe_qk_norm_after_init_(olmoe: OlmoeForCausalLM):
    """
    仅在 OLMoE 初始化后，重置每层 attention 的 q_norm/k_norm“模块本身”，
    使其形状为 OlmoeRMSNorm(head_dim)。

    只做模块替换，不改其它逻辑。
    """
    # 取到 OlmoeRMSNorm 类（从已构建的模型里拿，避免额外 import/版本差异）
    NormCls = olmoe.model.layers[0].self_attn.q_norm.__class__

    with torch.no_grad():
        for layer in olmoe.model.layers:
            attn = layer.self_attn
            head_dim = attn.head_dim
            eps = olmoe.config.rms_norm_eps

            # 保持 device / dtype 与原模块一致
            device = attn.q_norm.weight.device
            dtype = attn.q_norm.weight.dtype

            attn.q_norm = NormCls(head_dim, eps=eps).to(device=device, dtype=dtype)

            device = attn.k_norm.weight.device
            dtype = attn.k_norm.weight.dtype
            attn.k_norm = NormCls(head_dim, eps=eps).to(device=device, dtype=dtype)


# ---------------------- 主转换函数 ---------------------- #

def convert_qwen3_to_olmoe_local(
    qwen3_model_path: str,
    save_directory: str,
    num_local_experts: int = 8,
    num_experts_per_tok: int = 2,
    router_aux_loss_coef: float = 0.001,
    device: str = "cpu",
    dtype: str = "float16",
    noise_std: float = 0.02,
    noise_fraction: float = 0.5,
    router_init_strategy: str = "zeros",
):
    print(f"[1/4] 从本地加载 Qwen3ForCausalLM: {qwen3_model_path}")
    qwen = Qwen3ForCausalLM.from_pretrained(
        qwen3_model_path,
        torch_dtype=getattr(torch, dtype),
        device_map=None,
    )
    qwen.to(device)
    q_config: Qwen3Config = qwen.config

    # 统计 Qwen3 参数量（总 & 激活）
    q_total = count_parameters(qwen)
    q_active = estimate_qwen_activation_params(qwen)
    print("  Qwen3 模型参数统计：")
    print(f"    总参数量           : {format_param_count(q_total)}")
    print(f"    激活参数量(≈总)    : {format_param_count(q_active)}")

    print("[2/4] 根据 Qwen3 config 构造 OlmoeConfig")
    o_config = build_olmoe_config_from_qwen3(
        q_config,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
        router_aux_loss_coef=router_aux_loss_coef,
    )

    # Sanity check
    assert q_config.hidden_size == o_config.hidden_size
    assert q_config.num_attention_heads == o_config.num_attention_heads
    assert q_config.num_key_value_heads == o_config.num_key_value_heads
    assert q_config.intermediate_size == o_config.intermediate_size
    assert q_config.num_hidden_layers == o_config.num_hidden_layers

    print("[3/4] 创建随机初始化的 OlmoeForCausalLM（本地）")
    olmoe = OlmoeForCausalLM(o_config)
    olmoe.to(device)

    # ====== 按你的要求：OLMoE 初始化后，替换每层 q/k norm 模块为 OlmoeRMSNorm(head_dim) ======
    reinit_olmoe_qk_norm_after_init_(olmoe)
    # =====================================================================================

    # 统计 OLMoE 参数量（总 & 激活估算）
    o_total = count_parameters(olmoe)
    o_active = estimate_olmoe_activation_params(olmoe)
    print("  OLMoE 模型参数统计（转换后）：")
    print(f"    总参数量           : {format_param_count(o_total)}")
    print(f"    激活参数量(估算)   : {format_param_count(o_active)}")

    # 对比
    print("  参数量对比（OLMoE / Qwen3）：")
    print(f"    总参数比           : {o_total / q_total:.3f}x")
    print(f"    激活参数比         : {o_active / q_active:.3f}x")

    # 3.1 拷贝 embedding / lm_head
    print("  - 拷贝 embeddings 和 lm_head")
    copy_embedding_and_lm_head(qwen, olmoe)

    # 3.2 拷贝每一层：attention + MoE MLP + LayerNorm + Router 初始化
    print("  - 拷贝每一层（attention + MoE + layernorm + router init）")
    for idx, (q_layer, o_layer) in enumerate(zip(qwen.model.layers, olmoe.model.layers)):
        print(f"    * Layer {idx}")
        copy_attention_layer(q_layer, o_layer, idx, q_config, o_config)

        copy_mlp_to_moe_experts(
            q_layer,
            o_layer,
            idx,
            q_config,
            o_config,
            noise_std=noise_std,
            noise_fraction=noise_fraction,
        )

        copy_layernorms(q_layer, o_layer, idx)

        # Router 初始化：当前实现中 router 是 OlmoeSparseMoeBlock.gate
        router_module = o_layer.mlp.gate
        init_router(
            router_module,
            strategy=router_init_strategy,
            q_layer=q_layer,
            o_config=o_config,
            layer_idx=idx,
        )

    # 3.3 拷贝最末尾 norm
    print("  - 拷贝 final RMSNorm")
    copy_final_norm(qwen, olmoe)

    # [4/4] 保存
    print(f"[4/4] 保存转换后的 OLMoE 模型到: {save_directory}")
    olmoe.save_pretrained(save_directory)
    o_config.save_pretrained(save_directory)
    print("完成。")


# ---------------------- CLI ---------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3_path", type=str, required=True, help="本地 Qwen3ForCausalLM 目录")
    parser.add_argument("--save_dir", type=str, required=True, help="保存初始化后 OLMoE 模型的目录")
    parser.add_argument("--num_local_experts", type=int, default=8, help="每层本地 expert 数（num_local_experts）")
    parser.add_argument("--num_experts_per_tok", type=int, default=2, help="每个 token 的 top-k expert 数")
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.001, help="router 辅助 loss 系数")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    parser.add_argument("--dtype", type=str, default="float16", help="float16 / bfloat16 / float32")

    # Random Noise Upcycling 参数
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0,
        help="Random Noise Upcycling: 高斯噪声标准差，例如 0.02",
    )
    parser.add_argument(
        "--noise_fraction",
        type=float,
        default=0,
        help="每个 FFN 权重中加噪声的比例，如 0.5 表示 50%",
    )

    # Router 初始化策略
    parser.add_argument(
        "--router_init_strategy",
        type=str,
        default="zeros",
        choices=["zeros", "normal", "scaled_normal", "qwen_mlp_based"],
        help="router 参数初始化策略",
    )

    args = parser.parse_args()

    convert_qwen3_to_olmoe_local(
        qwen3_model_path=args.qwen3_path,
        save_directory=args.save_dir,
        num_local_experts=args.num_local_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        router_aux_loss_coef=args.router_aux_loss_coef,
        device=args.device,
        dtype=args.dtype,
        noise_std=args.noise_std,
        noise_fraction=args.noise_fraction,
        router_init_strategy=args.router_init_strategy,
    )


if __name__ == "__main__":
    main()