# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
将 HF Transformers 中的 OlmoeForCausalLM 权重转换为原 OLMoE 代码所使用的
(model.pt + config.yaml) 格式。

用法示例：

python convert_hf_olmoe_to_olmoe.py \
  --hf_model_dir ./olmoe_hf \
  --output_dir /data/llm/checkpoints/olmoe-from-hf \
  --config_yaml /data/llm/checkpoints/original_olmoe/config.yaml

说明：
- --hf_model_dir：你通过 `OlmoeForCausalLM.from_pretrained` 能加载的 HF 模型目录
- --output_dir：输出目录，会生成 model.pt（以及可选 config.yaml）
- --config_yaml（可选）：
    - 如果提供：直接把这个 config.yaml 拷贝到 output_dir
    - 如果不提供：脚本会从 HF config 中尽量推回一个 config.yaml（字段和你原脚本匹配）
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
import yaml
from typing import Optional
from transformers import OlmoeForCausalLM, OlmoeConfig
import torch.nn as nn
from transformers.models.olmoe.modeling_olmoe import OlmoeRMSNorm, OlmoeAttention
def patched_init(self, config, layer_idx: Optional[int] = None):
    super(OlmoeAttention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(
        config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
    )
    self.k_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
    )
    self.v_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
    )

    # 你要改的就是这两行
    self.q_norm = OlmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.k_norm = OlmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)

OlmoeAttention.__init__ = patched_init


def infer_olmoe_config_from_hf(hf_config: OlmoeConfig) -> dict:
    """
    根据 HF 的 OlmoeConfig 反推一个与你原始 OLMoE 训练代码兼容的 config 字典。
    字段名是参照你 forward 转换脚本中读取的 `olmoe_config["..."]` 来写的。
    如有不匹配，可在此处调整。
    """
    d_model = hf_config.hidden_size
    n_layers = hf_config.num_hidden_layers
    n_heads = hf_config.num_attention_heads
    vocab_size = hf_config.vocab_size

    # 这是 MoE 相关设置，需要和你训练时保持一致
    num_experts = hf_config.num_experts
    moe_top_k = hf_config.num_experts_per_tok
    intermediate_size = hf_config.intermediate_size  # 每个 expert 的维度，你原脚本里是 dim_per_expert

    # 注意：有些字段 HF config 中未直接暴露，视你实际需要加默认值或从别处推导
    olmoe_config = {
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "vocab_size": vocab_size,
        # 下面这些字段名是你原脚本访问过的
        "rope_theta": getattr(hf_config, "rope_theta", 10000.0),
        "max_sequence_length": getattr(hf_config, "max_position_embeddings", 2048),
        "pad_token_id": hf_config.pad_token_id,
        "eos_token_id": hf_config.eos_token_id,
        "weight_tying": getattr(hf_config, "tie_word_embeddings", True),

        # MoE 相关
        "moe_top_k": moe_top_k,
        # 以下两个字段不是你前向脚本显式用到的，但通常会存在于原 config.yaml 中，
        # 如果你有更准确的配置来源，请在这里替换。
        "num_experts": num_experts,
        "ffn_dim": intermediate_size,

        # KV-head / MQA / GQA
        "n_kv_heads": getattr(hf_config, "num_key_value_heads", None),
        # 如果你原 config 里用的是 multi_query_attention 这个布尔标志，可以简单根据 n_kv_heads==1 判断
        "multi_query_attention": getattr(
            hf_config, "multi_query_attention", (getattr(hf_config, "num_key_value_heads", None) == 1)
        ),

        # 其它可能用到的字段，按需补充：
        "embedding_size": getattr(hf_config, "vocab_size", vocab_size),
        "clip_qkv": getattr(hf_config, "clip_qkv", None),
    }

    return {"model": olmoe_config}


def save_config_yaml(hf_config: OlmoeConfig, output_dir: Path, src_config_yaml: str = None):
    """
    - 若提供了 src_config_yaml：直接拷贝过去
    - 否则根据 hf_config 生成一个新的 config.yaml
    """
    output_config_path = output_dir / "config.yaml"

    if src_config_yaml is not None:
        src = Path(src_config_yaml)
        if not src.is_file():
            raise FileNotFoundError(f"--config_yaml 指定的文件不存在: {src}")
        shutil.copy(src, output_config_path)
        print(f"Copied existing config.yaml from {src} to {output_config_path}")
        return

    # 从 HF 推一个 config.yaml
    cfg_dict = infer_olmoe_config_from_hf(hf_config)
    with open(output_config_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote inferred config.yaml to {output_config_path}")


def convert_hf_to_olmoe(hf_model_dir: str, output_dir: str, config_yaml: str = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HF OlmoeForCausalLM from {hf_model_dir}")
    model = OlmoeForCausalLM.from_pretrained(hf_model_dir, torch_dtype=torch.float32, device_map="cpu")
    hf_config: OlmoeConfig = model.config

    print("Converting HF state_dict → OLMoE model.pt 格式")
    sd = model.state_dict()

    # 读取一些基础超参（用于 sanity check）
    n_layers = hf_config.num_hidden_layers
    n_heads = hf_config.num_attention_heads
    d_model = hf_config.hidden_size

    # KV heads 数量：GQA/MQA 兼容
    if getattr(hf_config, "num_key_value_heads", None) is not None:
        num_kv_heads = hf_config.num_key_value_heads
    else:
        # 如果没这个字段，就默认所有头都是独立的
        num_kv_heads = n_heads

    dims_per_head = d_model // n_heads
    fused_qkv_out_dim = d_model + dims_per_head * num_kv_heads * 2

    # 目标：构造一个跟你原始脚本相反的 state_dict：
    #   transformer.blocks.{layer}.att_proj.weight        (fused [q; k; v])
    #   transformer.blocks.{layer}.attn_out.weight
    #   transformer.blocks.{layer}.q_norm.weight
    #   transformer.blocks.{layer}.k_norm.weight
    #   transformer.blocks.{layer}.ffn.router.layer.weight
    #   transformer.blocks.{layer}.ff_norm.weight
    #   transformer.blocks.{layer}.attn_norm.weight
    #   transformer.blocks.{layer}.ffn.experts.mlp.expert_w1.{expert}
    #   transformer.blocks.{layer}.ffn.experts.mlp.expert_v1.{expert}
    #   transformer.blocks.{layer}.ffn.experts.mlp.expert_w2.{expert}
    #
    # 以及最后的：
    #   transformer.wte.weight
    #   transformer.ff_out.weight
    #   transformer.ln_f.weight

    new_sd = {}

    # --- embedding / final heads ---
    # 你的正向脚本中：
    #   "model.embed_tokens.weight" -> "transformer.wte.weight"
    #   "lm_head.weight"            -> "transformer.ff_out.weight"
    #   "model.norm.weight"         -> "transformer.ln_f.weight"
    new_sd["transformer.wte.weight"] = sd["model.embed_tokens.weight"].clone()
    new_sd["transformer.ff_out.weight"] = sd["lm_head.weight"].clone()
    new_sd["transformer.ln_f.weight"] = sd["model.norm.weight"].clone()

    # --- 每一层 ---
    for layer_i in range(n_layers):
        # HF -> 自定义
        # HF:
        #   model.layers.{i}.self_attn.q_proj.weight
        #   model.layers.{i}.self_attn.k_proj.weight
        #   model.layers.{i}.self_attn.v_proj.weight
        #   model.layers.{i}.self_attn.o_proj.weight
        #   model.layers.{i}.self_attn.q_norm.weight
        #   model.layers.{i}.self_attn.k_norm.weight
        #   model.layers.{i}.mlp.gate.weight (router)
        #   model.layers.{i}.input_layernorm.weight
        #   model.layers.{i}.post_attention_layernorm.weight
        #
        # 正向脚本中是：
        #   q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
        #       loaded[f"transformer.blocks.{i}.att_proj.weight"], fused_dims, dim=0
        #   )
        #   self_attn.o_proj.weight <- transformer.blocks.{i}.attn_out.weight
        #   self_attn.q_norm.weight <- transformer.blocks.{i}.q_norm.weight
        #   self_attn.k_norm.weight <- transformer.blocks.{i}.k_norm.weight
        #   mlp.gate.weight         <- transformer.blocks.{i}.ffn.router.layer.weight
        #   input_layernorm.weight  <- transformer.blocks.{i}.attn_norm.weight
        #   post_attention_layernorm.weight <- transformer.blocks.{i}.ff_norm.weight
        #
        # MoE experts:
        #   mlp.experts.{e}.gate_proj.weight -> transformer.blocks.{i}.ffn.experts.mlp.expert_w1.{e}
        #   mlp.experts.{e}.up_proj.weight   -> transformer.blocks.{i}.ffn.experts.mlp.expert_v1.{e}
        #   mlp.experts.{e}.down_proj.weight -> transformer.blocks.{i}.ffn.experts.mlp.expert_w2.{e}.T

        prefix = f"model.layers.{layer_i}"

        q = sd[f"{prefix}.self_attn.q_proj.weight"]
        k = sd[f"{prefix}.self_attn.k_proj.weight"]
        v = sd[f"{prefix}.self_attn.v_proj.weight"]

        # sanity check
        assert q.shape[0] == d_model, f"Unexpected q_proj out dim: {q.shape}"
        assert k.shape[0] == dims_per_head * num_kv_heads, f"Unexpected k_proj out dim: {k.shape}"
        assert v.shape[0] == dims_per_head * num_kv_heads, f"Unexpected v_proj out dim: {v.shape}"

        # 拼回 fused att_proj.weight
        att_proj_weight = torch.cat([q, k, v], dim=0)  # dim=0 对应你的正向脚本
        assert att_proj_weight.shape[0] == fused_qkv_out_dim

        new_sd[f"transformer.blocks.{layer_i}.att_proj.weight"] = att_proj_weight
        new_sd[f"transformer.blocks.{layer_i}.attn_out.weight"] = sd[f"{prefix}.self_attn.o_proj.weight"].clone()
        new_sd[f"transformer.blocks.{layer_i}.q_norm.weight"] = sd[f"{prefix}.self_attn.q_norm.weight"].clone()
        new_sd[f"transformer.blocks.{layer_i}.k_norm.weight"] = sd[f"{prefix}.self_attn.k_norm.weight"].clone()

        # Router / layernorm
        new_sd[f"transformer.blocks.{layer_i}.ffn.router.layer.weight"] = sd[f"{prefix}.mlp.gate.weight"].clone()
        new_sd[f"transformer.blocks.{layer_i}.attn_norm.weight"] = sd[f"{prefix}.input_layernorm.weight"].clone()
        new_sd[f"transformer.blocks.{layer_i}.ff_norm.weight"] = sd[f"{prefix}.post_attention_layernorm.weight"].clone()

        # MoE experts
        # 从任意一个 expert 的 down_proj 推出专家数和维度
        expert_0_down = sd[f"{prefix}.mlp.experts.0.down_proj.weight"]
        dim_per_expert = expert_0_down.shape[1]  # 因为正向中保存时是 .T，所以这里列数是 expert dim

        # 找出 num_experts：可以从 router.weight.shape[0] 来
        router_w = new_sd[f"transformer.blocks.{layer_i}.ffn.router.layer.weight"]
        num_experts = router_w.shape[0]

        for expert_i in range(num_experts):
            e_prefix = f"{prefix}.mlp.experts.{expert_i}"
            gate_proj = sd[f"{e_prefix}.gate_proj.weight"]
            up_proj = sd[f"{e_prefix}.up_proj.weight"]
            down_proj = sd[f"{e_prefix}.down_proj.weight"]

            # 正向脚本中：
            #   expert_w1.{e} = gate_proj.weight
            #   expert_v1.{e} = up_proj.weight
            #   expert_w2.{e} = down_proj.weight.T
            new_sd[f"transformer.blocks.{layer_i}.ffn.experts.mlp.expert_w1.{expert_i}"] = gate_proj.clone()
            new_sd[f"transformer.blocks.{layer_i}.ffn.experts.mlp.expert_v1.{expert_i}"] = up_proj.clone()
            new_sd[f"transformer.blocks.{layer_i}.ffn.experts.mlp.expert_w2.{expert_i}"] = down_proj.T.contiguous()

        # rotary_emb.inv_freq 在你的正向脚本是从 config 算出来的，没有存进 model.pt，
        # 所以这里不必再从 HF 中写回，除非你的原生训练代码需要在 checkpoint 中也存这个参数。
        # 如果需要，可以在这里从 HF 取到（model.layers.{i}.self_attn.rotary_emb.inv_freq）
        # 然后对应写回 new_sd。

    # 保存 model.pt
    out_model_path = output_dir / "model.pt"
    torch.save(new_sd, out_model_path)
    print(f"Saved converted OLMoE checkpoint to {out_model_path}")

    # 保存/复制 config.yaml
    save_config_yaml(hf_config, output_dir, src_config_yaml=config_yaml)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_dir",
        required=True,
        help="HF OlmoeForCausalLM 模型所在目录（即 from_pretrained 能加载的目录）",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="输出目录，将生成 model.pt（以及 config.yaml）",
    )
    parser.add_argument(
        "--config_yaml",
        default=None,
        help="可选：现有的原始 OLMoE config.yaml 路径；若提供，则直接拷贝到 output_dir。",
    )
    args = parser.parse_args()

    convert_hf_to_olmoe(
        hf_model_dir=args.hf_model_dir,
        output_dir=args.output_dir,
        config_yaml=args.config_yaml,
    )


if __name__ == "__main__":
    main()