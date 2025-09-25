models=("llama" "deepseek" "gpt" "Qwen")
models_2=("llama" "gpt_4o" "gpt_5" "gpt_5_thinking" "Qwen" "Qwen_thinking" "deepseek" "deepseek_thinking")
TRAIN_METHOD="DDL"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="None"
n_sample=2000
num_epochs=4

export WANDB_MODE=offline
#for model in ${models[@]}; do
#  accelerate launch detectors/detectanyllm/train.py \
#      --scoring_model_name ${SCORING_MODEL} \
#      --reference_model_name ${REFERENCE_MODEL} \
#      --wandb True \
#      --task thinking\
#      --train_data_path ./data/v1/thinking/${model}_sample.csv \
#      --train_data_format detect \
#      --save_directory ./detectors/detectanyllm/ckpt/v1/thinking/ \
#      --wandb_dir ./detectors/detectanyllm/log/v1/thinking/ \
#      --train_method ${TRAIN_METHOD} \
#      --n_sample ${n_sample} \
#      --num_epochs ${num_epochs}
#done

## Eval
for model in "${models_2[@]}"; do
  if [[ "$model" == "llama" ]]; then
      base_model="llama"
  elif [[ "$model" == "deepseek" || "$model" == "deepseek_thinking" ]]; then
      base_model="deepseek"
  elif [[ "$model" == "gpt_4o" || "$model" == "gpt_5" || "$model" == "gpt_5_thinking" ]]; then
      base_model="gpt"
  elif [[ "$model" == "Qwen" || "$model" == "Qwen_thinking" ]]; then
      base_model="Qwen"
  fi
  accelerate launch detectors/detectanyllm/eval.py \
  --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/thinking/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${base_model}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
  --eval_data_path ./data/v1/thinking/${model}_sample.csv \
  --eval_data_format detect \
  --wandb True \
  --train_method ${TRAIN_METHOD} \
  --eval_batch_size 4 \
  --save_dir ./detectors/detectanyllm/results/v1/thinking \
  --save_file ${model}_n${n_sample}_multi${multilen}.json
done