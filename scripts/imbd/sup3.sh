datasets=("llama" "deepseek" "gpt4o" "Qwen")
frac="0.2"
device='cuda:3'
DATANUM=2000
EPOCHS=4
LR=1e-4
BETA=0.05

# cross-model
#for dataset in "${datasets[@]}"; do
#  if [ "$dataset" = "llama" ]; then
#    model2="deepseek"
#  elif [ "$dataset" = "deepseek" ]; then
#    model2="llama"
#  elif [ "$dataset" = "gpt4o" ]; then
#    model2="llama"
#  elif [ "$dataset" = "Qwen" ]; then
#    model2="llama"
#  fi
#  echo "Running with model: $dataset"
#  python detectors/imbd/run_spo.py \
#    --datanum=$DATANUM \
#    --task_name llm-co \
#    --epochs=$EPOCHS \
#    --lr=$LR \
#    --beta=$BETA \
#    --device=$device \
#    --train_dataset "$dataset" \
#    --eval_dataset "$dataset" \
#    --frac=$frac \
#    --model2 $model2
#done

## eval
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
  for model2 in "${model2s[@]}"; do
    echo "Processing cross-domain: $dataset, model2: $model2, flag 1"
    python detectors/imbd/run_spo.py \
      --eval_only True \
      --from_pretrained "./detectors/imbd/ckpt/v1/llm-co/${dataset}_n2000_ep4_spo_lr_0.0001_beta_0.05_a_1" \
      --train_dataset "$dataset" \
      --eval_dataset "$dataset" \
      --device $device \
      --task_name llm-co \
      --frac $frac \
      --model2 $model2 \
      --flag "1"
  done

    for model2 in "${model2s[@]}"; do
    echo "Processing cross-domain: $dataset, model2: $model2, flag 2"
    python detectors/imbd/run_spo.py \
      --eval_only True \
      --from_pretrained "./detectors/imbd/ckpt/v1/llm-co/${dataset}_n2000_ep4_spo_lr_0.0001_beta_0.05_a_1" \
      --train_dataset "$dataset" \
      --eval_dataset "$dataset" \
      --device $device \
      --task_name llm-co \
      --frac $frac \
      --model2 $model2 \
      --flag "2"
  done
done