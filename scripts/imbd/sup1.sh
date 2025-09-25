models=("llama" "deepseek" "gpt_4o" "Qwen")
models_2=("llama" "gpt_4o" "gpt_5" "gpt_5_thinking" "Qwen" "Qwen_thinking" "deepseek" "deepseek_thinking")
DATANUM=2000
n_sample=2000
EPOCHS=4
LR=1e-4
BETA=0.05
device='cuda:1'

# cross-model
#for MODEL in "${models[@]}"; do
#  echo "Running $TASK_NAME with model: $MODEL"
#  python detectors/imbd/run_spo.py \
#    --datanum=$DATANUM \
#    --task_name thinking \
#    --epochs=$EPOCHS \
#    --lr=$LR \
#    --beta=$BETA \
#    --device=$device \
#    --train_dataset "$MODEL" \
#    --eval_dataset "$MODEL" \
#    --n_sample=$n_sample
#done

## eval
for MODEL in ${models_2[@]}; do
  if [[ "$MODEL" == "llama" ]]; then
      base_model="llama"
  elif [[ "$MODEL" == "deepseek" || "$model" == "deepseek_thinking" ]]; then
      base_model="deepseek"
  elif [[ "$MODEL" == "gpt_4o" || "$model" == "gpt_5" || "$model" == "gpt_5_thinking" ]]; then
      base_model="gpt_4o"
  elif [[ "$MODEL" == "Qwen" || "$model" == "Qwen_thinking" ]]; then
      base_model="Qwen"
  fi
  echo "Processing cross-model: $MODEL"
  python detectors/imbd/run_spo.py \
    --eval_only True \
    --from_pretrained "./detectors/imbd/ckpt/v1/thinking/${base_model}_n${n_sample}_ep${EPOCHS}_spo_lr_0.0001_beta_0.05_a_1" \
    --eval_dataset "$MODEL" \
    --device $device \
    --task_name thinking
done