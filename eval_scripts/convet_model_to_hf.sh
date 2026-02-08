RUN_DIR=$1
SAVE_STEP=$2
MODEL_SIZE=$3
TAG=$4

python scripts/convert_olmoe_custom_to_hf.py \
    --input_dir ${RUN_DIR}/step${SAVE_STEP}-unsharded \
    --tokenizer_path Qwen/Qwen2.5-0.5B \
    --output_dir hf_models/SPES-${MODEL_SIZE}-step${SAVE_STEP}-${TAG}-hf \
    --no_fix_eos_token_id \