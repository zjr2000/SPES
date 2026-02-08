# Copyright 2024 The HuggingFace Inc. team and Contributors.
#
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

import argparse
import os
import torch
import yaml
import shutil
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

"""
Script to convert a Hugging Face Qwen2/Qwen2.5 model to the custom flat format 
(model.pt + config.yaml) compatible with the provided OLMo-style loader.

Usage:
python convert_hf_qwen_to_custom.py --model_id Qwen/Qwen2.5-7B --output_dir ./my_custom_weights
"""

def save_config(hf_config, output_dir):
    """
    Maps HF Qwen2 config to the YAML format expected by the custom loader.
    """
    # Calculate intermediate size if not explicitly present, though Qwen usually has it.
    # Qwen2 uses intermediate_size directly.
    
    config_dict = {
        "d_model": hf_config.hidden_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_heads": hf_config.num_attention_heads,
        "n_kv_heads": hf_config.num_key_value_heads,
        "max_sequence_length": hf_config.max_position_embeddings,
        "vocab_size": hf_config.vocab_size,
        "embedding_size": hf_config.vocab_size,
        "mlp_hidden_size": hf_config.intermediate_size * 2, # The loader divides this by 2
        "mlp_ratio": None, # Using explicit size instead
        "pad_token_id": hf_config.pad_token_id,
        "eos_token_id": hf_config.eos_token_id,
        "weight_tying": hf_config.tie_word_embeddings,
        "clip_qkv": None,
        "rope_theta": hf_config.rope_theta,
        "model_type": "qwen2" # Metadata
    }

    # Wrap in 'model' key as expected by the loader
    final_config = {"model": config_dict}
    
    # Also save tokenizer info if possible
    final_config["tokenizer"] = {
        "identifier": "tokenizer.json",
        "vocab_size": hf_config.vocab_size
    }

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(final_config, f, default_flow_style=False)
    print(f"Configuration saved to {os.path.join(output_dir, 'config.yaml')}")

def convert_weights(model, output_dir):
    """
    Converts HF state_dict to the monolithic model.pt format.
    """
    print("Loading HF model weights into memory...")
    hf_state_dict = model.state_dict()
    custom_state_dict = {}

    print("Converting layers...")
    num_layers = model.config.num_hidden_layers

    for i in range(num_layers):
        # --- Attention ---
        # HF Qwen: q_proj, k_proj, v_proj are separate.
        # Custom Loader expects: transformer.blocks.{i}.att_proj.weight containing [Q, K, V]
        q = hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        k = hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        v = hf_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        
        # Concatenate Q, K, V along dim 0
        custom_state_dict[f"transformer.blocks.{i}.att_proj.weight"] = torch.cat([q, k, v], dim=0)
        custom_state_dict[f"transformer.blocks.{i}.q_norm.weight"] = hf_state_dict[f"model.layers.{i}.self_attn.q_norm.weight"]
        custom_state_dict[f"transformer.blocks.{i}.k_norm.weight"] = hf_state_dict[f"model.layers.{i}.self_attn.k_norm.weight"]

        # Output projection
        custom_state_dict[f"transformer.blocks.{i}.attn_out.weight"] = hf_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"]

        # --- MLP ---
        # HF Qwen: gate_proj, up_proj, down_proj
        # Custom Loader expects: transformer.blocks.{i}.ff_proj.weight containing [Up, Gate]
        # Note: The provided loader script does: up, gate = torch.chunk(weight, 2).
        # So we must concatenate [Up, Gate].
        up = hf_state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        gate = hf_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        
        custom_state_dict[f"transformer.blocks.{i}.ff_proj.weight"] = torch.cat([up, gate], dim=0)
        
        # Down projection
        custom_state_dict[f"transformer.blocks.{i}.ff_out.weight"] = hf_state_dict[f"model.layers.{i}.mlp.down_proj.weight"]

        # --- Layer Norms ---
        # The provided loader script had "# TODO: Layernorm stuff", so we map them to standard names.
        # You may need to adjust these keys if your custom inference code names them differently.
        custom_state_dict[f"transformer.blocks.{i}.attn_norm.weight"] = hf_state_dict[f"model.layers.{i}.input_layernorm.weight"]
        custom_state_dict[f"transformer.blocks.{i}.ff_norm.weight"] = hf_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"]

    # --- Embeddings & Head ---
    print("Converting embeddings and head...")
    custom_state_dict["transformer.wte.weight"] = hf_state_dict["model.embed_tokens.weight"]
    
    # Qwen usually ties weights, but if lm_head exists separately, use it.
    if "lm_head.weight" in hf_state_dict:
        custom_state_dict["transformer.ff_out.weight"] = hf_state_dict["lm_head.weight"]
    else:
        # Fallback if tied
        custom_state_dict["transformer.ff_out.weight"] = hf_state_dict["model.embed_tokens.weight"]

    # Final Layer Norm
    custom_state_dict["transformer.ln_f.weight"] = hf_state_dict["model.norm.weight"]

    # Save
    save_path = os.path.join(output_dir, "model.pt")
    print(f"Saving monolithic checkpoint to {save_path}...")
    torch.save(custom_state_dict, save_path)
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        required=True,
        help="Hugging Face model ID (e.g., Qwen/Qwen2.5-7B) or local path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the converted model.pt and config.yaml.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HF cache directory.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_id}")
    # Load on CPU to avoid OOM during conversion of large models
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.float32, 
        device_map="cpu", 
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Convert Config
    save_config(model.config, args.output_dir)

    # Convert Weights
    convert_weights(model, args.output_dir)

    # Save Tokenizer (simply copying the tokenizer.json usually works for these custom loaders)
    print("Saving tokenizer...")
    tokenizer.save_pretrained(args.output_dir)
    
    # Ensure a tokenizer.json exists for the loader to find
    if not os.path.exists(os.path.join(args.output_dir, "tokenizer.json")):
        # Some HF tokenizers save as tokenizer.model or special JSONs. 
        # We try to ensure the standard json exists.
        try:
            tokenizer.save_vocabulary(args.output_dir)
        except Exception as e:
            print(f"Warning: Could not save vocabulary explicitly: {e}")

if __name__ == "__main__":
    main()