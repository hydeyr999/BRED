models=("llama" "deepseek" "gpt4o" "Qwen")
models_2=('llama' 'llama_8b_instruct' 'deepseek' 'gpt4o' 'gpt4o_large' 'Qwen' 'Qwen_7b' 'Qwen_8b')
num_epochs=4
threshold="0.0"
n_sample=2000
TRAIN_METHOD="DDL"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="None"

export WANDB_MODE=offline
#for model in ${models[@]}; do
#  accelerate launch detectors/detectanyllm/train.py \
#      --scoring_model_name ${SCORING_MODEL} \
#      --reference_model_name ${REFERENCE_MODEL} \
#      --wandb True \
#      --task cross-model\
#      --train_data_path ./data/v1/rebuttal/${model}_sample.csv \
#      --train_data_format detect \
#      --save_directory ./detectors/detectanyllm/ckpt/v1/cross-model/ \
#      --wandb_dir ./detectors/detectanyllm/log/v1/cross-model/ \
#      --train_method ${TRAIN_METHOD} \
#      --n_sample ${n_sample} \
#      --num_epochs ${num_epochs} \
#      --re True
#done

for model in "${models_2[@]}"; do
  if [[ "$model" == "llama" || "$model" == "llama_8b_instruct" ]]; then
      base_model="llama"
  elif [[ "$model" == "deepseek" ]]; then
      base_model="deepseek"
  elif [[ "$model" == "gpt4o" || "$model" == "gpt4o_large" ]]; then
      base_model="gpt4o"
  elif [[ "$model" == "Qwen" || "$model" == "Qwen_7b" || "$model" == "Qwen_8b" ]]; then
      base_model="Qwen"
  fi
  accelerate launch detectors/detectanyllm/eval.py \
  --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/cross-model/re/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${base_model}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
  --eval_data_path ./data/v1/rebuttal/${model}_sample.csv \
  --eval_data_format detect \
  --wandb True \
  --train_method ${TRAIN_METHOD} \
  --eval_batch_size 4 \
  --save_dir ./detectors/detectanyllm/results/v1/cross-model \
  --save_file ${model}_n${n_sample}_multi${multilen}.json
done

