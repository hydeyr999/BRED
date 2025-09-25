datasets=("llama" "deepseek" "gpt4o" "Qwen")
frac="0.2"
epochs=4
threshold="0.0"
device='cuda:1'

## Train
#for dataset in ${datasets[@]}; do
#  echo "Training on $dataset, epochs=$epochs, threshold=$threshold"
#  python detectors/deberta/run_deberta.py \
#  --task llm-co \
#  --dataset $dataset \
#  --mode train \
#  --device $device \
#  --epochs $epochs \
#  --threshold $threshold \
#  --frac $frac
#done


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
    echo "Evaluating on $dataset, epochs=$epochs, threshold=$threshold, model2=$model2"
    python detectors/deberta/run_deberta.py \
    --task llm-co \
    --dataset $dataset \
    --mode test \
    --deberta_model ./detectors/deberta/weights/v1/llm-co/deberta_$dataset\_ep$epochs\_thres$threshold\_n$frac \
    --device $device \
    --epochs $epochs \
    --threshold $threshold \
    --frac $frac \
    --model2 $model2
  done
done