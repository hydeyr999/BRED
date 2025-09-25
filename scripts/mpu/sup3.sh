datasets=("llama" "deepseek" "gpt4o" "Qwen")
frac="0.2"
epochs=4
threshold="0.0"
device='cuda:2'

#for dataset in "${datasets[@]}"; do
#  CUDA_VISIBLE_DEVICES=0 python detectors/mpu/train.py \
#  --batch-size 4 \
#  --max-sequence-length 512 \
#  --task llm-co \
#  --dataset ${dataset} \
#  --model-name roberta-large \
#  --local-data data \
#  --lamb 0.4 \
#  --prior 0.2 \
#  --pu_type dual_softmax_dyn_dtrun \
#  --len_thres 55 \
#  --aug_min_length 1 \
#  --max-epochs $epochs \
#  --weight-decay 0 \
#  --mode original_single \
#  --aug_mode sentence_deletion-0.25 \
#  --clean 1 \
#  --learning-rate 5e-05 \
#  --seed 0 \
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
    echo "Testing dataset: $dataset, epochs: $epochs, threshold: $threshold, model2: $model2"
    python detectors/opensource/mpu.py \
    --task llm-co \
    --dataset $dataset \
    --device $device \
    --mpu_model ./detectors/mpu/results/v1/llm-co/mpu_${dataset}_n${frac}_ep$epochs\_thres$threshold/complete-${epochs} \
    --frac $frac \
    --model2 $model2
  done
done