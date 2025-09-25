datasets=("llama" "deepseek" "gpt4o" "Qwen")
frac="0.2"
device='cuda:3'

#for dataset in ${datasets[@]}; do
#  echo "Training on $dataset"
#  python detectors/ling-based/ghostbuster.py \
#  --task llm-co \
#  --dataset $dataset \
#  --device $device \
#  --mode train \
#  --frac $frac
#done

## Evaluate
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
    echo "Evaluating on $dataset, $model2, func=$func, flag=1"
    python detectors/ling-based/ghostbuster.py \
    --task llm-co \
    --dataset $dataset \
    --classifier ./detectors/ling-based/classifier/v1/llm-co/gb_$dataset\_n${frac}.pkl \
    --mode test \
    --frac $frac \
    --model2 $model2 \
    --flag "1"
  done

  for model2 in ${model2s[@]}; do
    echo "Evaluating on $dataset, $model2, func=$func, flag=2"
    python detectors/ling-based/ghostbuster.py \
    --task llm-co \
    --dataset $dataset \
    --classifier ./detectors/ling-based/classifier/v1/llm-co/gb_$dataset\_n${frac}.pkl \
    --mode test \
    --frac $frac \
    --model2 $model2 \
    --flag "2"
  done
done