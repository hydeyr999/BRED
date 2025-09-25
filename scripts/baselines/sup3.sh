datasets=("llama" "deepseek" "gpt4o" "Qwen")
frac="0.2"
device='cuda:2'

for dataset in ${datasets[@]}; do
  if [[ "$dataset" == "llama" ]]; then
    model2s=("deepseek" "gpt4o" "Qwen")
  elif [[ "$dataset" == "deepseek" ]]; then
    model2s=("llama" "gpt4o" "Qwen")
  elif [[ "$dataset" == "gpt4o" ]]; then
    model2s=("llama" "deepseek" "Qwen")
  elif [[ "$dataset" == "Qwen" ]]; then
    model2s=("llama" "deepseek" "gpt4o")
  fi
  for model2 in ${model2s[@]}; do
    echo "Processing $dataset on baselines..."
    python detectors/baselines/baselines.py \
    --task llm-co \
    --dataset $dataset \
    --frac $frac \
    --device $device \
    --model2 $model2
  done
done