#models=("gpt_5" "gpt_5_thinking" "deepseek_thinking" "Qwen_thinking")
models=("llama" "deepseek" "gpt" "Qwen")
models_2=("llama" "gpt_4o" "gpt_5" "gpt_5_thinking" "Qwen" "Qwen_thinking" "deepseek" "deepseek_thinking")
device="cuda:2"
epochs=4
threshold="0.0"
n_sample=2000

#for item in "${models[@]}"; do
#  echo "Generating dpic data for thinking dataset: $item"
#  python detectors/DPIC/run_dpic.py \
#    --run_generate True \
#    --task thinking \
#    --dataset $item \
#    --device $device
#done

#for model in ${models[@]}; do
#  echo "Training on $model, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
#  python detectors/DPIC/run_dpic.py \
#  --task thinking \
#  --dataset $model \
#  --mode train \
#  --device $device \
#  --epochs $epochs \
#  --threshold $threshold \
#  --n_sample $n_sample
#done

for model in ${models_2[@]}; do
  if [[ "$model" == "llama" ]]; then
        base_model="llama"
  elif [[ "$model" == "deepseek" || "$model" == "deepseek_thinking" ]]; then
      base_model="deepseek"
  elif [[ "$model" == "gpt_4o" || "$model" == "gpt_5" || "$model" == "gpt_5_thinking" ]]; then
      base_model="gpt"
  elif [[ "$model" == "Qwen" || "$model" == "Qwen_thinking" ]]; then
      base_model="Qwen"
  fi
  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/DPIC/run_dpic.py \
  --task thinking \
  --dataset $model \
  --mode test \
  --dpic_ckpt ./detectors/DPIC/weights/v1/thinking/dpic_${base_model}_ep4_thres0.0_n2000.pth \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample
done