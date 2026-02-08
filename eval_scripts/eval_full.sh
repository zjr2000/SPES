MODEL_PATH=$1
MODEL_NAME=$2

accelerate launch --main_process_port 20503 -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True \
    --tasks sciq,piqa,winogrande,arc_easy,social_iqa,openbookqa,boolq \
    --batch_size auto \
    --trust_remote_code \
    --output_path eval_output/${MODEL_NAME}

accelerate launch --main_process_port 20503 -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --batch_size auto \
    --trust_remote_code \
    --output_path eval_output/${MODEL_NAME}

accelerate launch --main_process_port 20503 -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True \
    --tasks hellaswag \
    --num_fewshot 10 \
    --batch_size auto \
    --trust_remote_code \
    --output_path eval_output/${MODEL_NAME}


accelerate launch --main_process_port 20503 -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True \
    --trust_remote_code \
    --tasks ceval-valid,cmmlu,mmlu \
    --batch_size auto \
    --output_path eval_output/${MODEL_NAME}
    

