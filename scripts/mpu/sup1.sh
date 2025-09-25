models=("llama" "deepseek" "gpt" "Qwen")
models_2=("llama" "gpt_4o" "gpt_5" "gpt_5_thinking" "Qwen" "Qwen_thinking" "deepseek" "deepseek_thinking")
n_sample=2000
epochs=4
threshold="0.0"
device="cuda:3"

#for model in "${models[@]}"; do
#  CUDA_VISIBLE_DEVICES=0 python detectors/mpu/train.py \
#  --batch-size 4 \
#  --max-sequence-length 512 \
#  --task thinking \
#  --dataset ${model} \
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
#  --n_sample $n_sample \
#  --threshold $threshold
#done

## Eval
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
  python detectors/opensource/mpu.py \
  --task thinking \
  --dataset $model \
  --device $device \
  --mpu_model ./detectors/mpu/results/v1/thinking/mpu_${base_model}_ep$epochs\_thres$threshold\_n$n_sample/complete-${epochs} \
  --n_sample $n_sample
done
