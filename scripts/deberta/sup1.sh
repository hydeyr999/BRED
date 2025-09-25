models=("llama" "deepseek" "gpt" "Qwen")
models_2=("llama" "gpt_4o" "gpt_5" "gpt_5_thinking" "Qwen" "Qwen_thinking" "deepseek" "deepseek_thinking")
epochs=4
threshold="0.0"
n_sample=2000
device='cuda:2'

#for model in ${models[@]}; do
#  echo "Training on $model, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
#  python detectors/deberta/run_deberta.py \
#  --task thinking \
#  --dataset $model \
#  --mode train \
#  --device $device \
#  --epochs $epochs \
#  --threshold $threshold \
#  --n_sample $n_sample
#done

## eval
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
  python detectors/deberta/run_deberta.py \
  --task thinking \
  --dataset $model \
  --mode test \
  --dpic_ckpt ./detectors/deberta/weights/v1/thinking/deberta_$base_model\_ep$epochs\_thres$threshold\_n$n_sample \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample
done