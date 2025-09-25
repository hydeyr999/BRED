models=("llama" "deepseek" "gpt4o" "Qwen")
#models_2=('llama' 'llama_8b_instruct' 'deepseek' 'gpt4o' 'gpt4o_large' 'Qwen' 'Qwen_7b' 'Qwen_8b')
models_2=('Qwen_8b')
epochs=4
threshold="0.0"
n_sample=2000
device="cuda:0"

#for model in "${models[@]}"; do
#  CUDA_VISIBLE_DEVICES=0 python detectors/mpu/train.py --batch-size 4 --max-sequence-length 512 \
#    --task cross-model --dataset ${model} --val-data-file v1/rebuttal/${model}_sample.csv \
#    --model-name roberta-large --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun \
#    --len_thres 55 --aug_min_length 1 --max-epochs $epochs --weight-decay 0 --mode original_single \
#    --aug_mode sentence_deletion-0.25 --clean 1 --learning-rate 5e-05 --seed 0 \
#    --n_sample $n_sample --threshold $threshold \
#    --re True
#done

for model in ${models_2[@]}; do
  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  if [[ "$model" == "llama" || "$model" == "llama_8b_instruct" ]]; then
      base_model="llama"
  elif [[ "$model" == "deepseek" ]]; then
      base_model="deepseek"
  elif [[ "$model" == "gpt4o" || "$model" == "gpt4o_large" ]]; then
      base_model="gpt4o"
  elif [[ "$model" == "Qwen" || "$model" == "Qwen_7b" || "$model" == "Qwen_8b" ]]; then
      base_model="Qwen"
  fi
  python detectors/opensource/mpu.py \
  --task cross-model \
  --dataset $model \
  --device $device \
  --mpu_model ./detectors/mpu/results/v1/cross-model/re/mpu_${base_model}_ep$epochs\_thres$threshold\_n$n_sample/complete-${epochs} \
  --n_sample $n_sample \
  --re True
done