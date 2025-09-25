datasets=("llama" "deepseek" "gpt4o" "Qwen")
device="cuda:2"

#for item in "${datasets[@]}"; do
#  echo "Generating dpic data for llm-co dataset: $item, train mode"
#  python detectors/DPIC/run_dpic.py \
#    --run_generate True \
#    --task llm-co \
#    --dataset $item \
#    --device $device \
#    --mode train
#done

#for item in "${datasets[@]}"; do
#  echo "Generating dpic data for llm-co dataset: $item, test mode"
#  python detectors/DPIC/run_dpic.py \
#    --run_generate True \
#    --task llm-co \
#    --dataset $item \
#    --device $device
#done

#for item in "${datasets[@]}"; do
#  echo "Generating dpic data for llm-co dataset: $item, train mode"
#  python detectors/DPIC/run_dpic.py \
#    --task llm-co \
#    --dataset $item \
#    --device $device \
#    --mode train
#done

for item in "${datasets[@]}"; do
  if [[ "$item" == "llama" ]]; then
    model2s=("deepseek" "gpt4o" "Qwen")
  elif [[ "$item" == "deepseek" ]]; then
    model2s=("llama" "gpt4o" "Qwen")
  elif [[ "$item" == "gpt4o" ]]; then
    model2s=("llama" "deepseek" "Qwen")
  elif [[ "$item" == "Qwen" ]]; then
    model2s=("llama" "deepseek" "gpt4o")
  fi

  for model2 in "${model2s[@]}"; do
    echo "Generating dpic data for llm-co dataset: $item, model2: $model2, flag 1"
    python detectors/DPIC/run_dpic.py \
      --task llm-co \
      --dataset $item \
      --dpic_ckpt ./detectors/DPIC/weights/v1/llm-co/dpic_$item\_ep4_thres0_n250.pth \
      --device $device \
      --model2 $model2 \
      --flag 1
  done

  for model2 in "${model2s[@]}"; do
    echo "Generating dpic data for llm-co dataset: $item, model2: $model2, flag 2"
    python detectors/DPIC/run_dpic.py \
      --task llm-co \
      --dataset $item \
      --dpic_ckpt ./detectors/DPIC/weights/v1/llm-co/dpic_$item\_ep4_thres0_n250.pth \
      --device $device \
      --model2 $model2 \
      --flag 2
  done
done