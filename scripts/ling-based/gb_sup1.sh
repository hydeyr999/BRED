models=("llama" "deepseek" "gpt" "Qwen")
models_2=("llama" "gpt_4o" "gpt_5" "gpt_5_thinking" "Qwen" "Qwen_thinking" "deepseek" "deepseek_thinking")
n_sample=2000
device='cuda:0'

#for model in ${models[@]}; do
#  echo "Training on $model, n_sample=$n_sample"
#  python detectors/ling-based/ghostbuster.py \
#  --task thinking \
#  --dataset $model \
#  --n_sample $n_sample \
#  --device $device \
#  --mode train
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
  echo "Testing on $model, n_sample=$n_sample"
  python detectors/ling-based/ghostbuster.py \
  --task thinking \
  --dataset $model \
  --mode test \
  --classifier ./detectors/ling-based/classifier/v1/thinking/gb_$base_model\_n${n_sample}.pkl \
  --n_sample $n_sample \
  --device $device
done