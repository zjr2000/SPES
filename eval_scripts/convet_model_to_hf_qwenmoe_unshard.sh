
RUN_DIR=$1
SAVE_STEP=$2
MODEL_SIZE=$3

python scripts/unshard_new.py ${RUN_DIR}/step${SAVE_STEP} hf_models/SPES-${MODEL_SIZE}-step${SAVE_STEP}

python scripts/convert_olmoe_custom_to_hf_qwen_moe.py \
    --input_dir hf_models/SPES-${MODEL_SIZE}-step${SAVE_STEP} \
    --tokenizer_path Qwen/Qwen2.5-0.5B \
    --output_dir hf_models/SPES-${MODEL_SIZE}-step${SAVE_STEP}-hf \
    --no_fix_eos_token_id \