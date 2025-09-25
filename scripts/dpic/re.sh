models=("llama" "deepseek" "gpt4o" "Qwen")
models_2=('llama' 'llama_8b_instruct' 'deepseek' 'gpt4o' 'gpt4o_large' 'Qwen' 'Qwen_7b' 'Qwen_8b')
epochs=4
threshold="0.0"
n_sample=2000
device="cuda:0"

#for model in ${models[@]}; do
#  echo "Training on $model, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
#  python detectors/DPIC/run_dpic.py \
#  --task rebuttal \
#  --dataset $model \
#  --mode train \
#  --device $device \
#  --epochs $epochs \
#  --threshold $threshold \
#  --n_sample $n_sample
#done

for model in ${models_2[@]}; do
  if [[ "$model" == "llama" || "$model" == "llama_8b_instruct" ]]; then
        base_model="llama"
  elif [[ "$model" == "deepseek" ]]; then
      base_model="deepseek"
  elif [[ "$model" == "gpt4o" || "$model" == "gpt4o_large" ]]; then
      base_model="gpt4o"
  elif [[ "$model" == "Qwen" || "$model" == "Qwen_7b" || "$model" == "Qwen_8b" ]]; then
      base_model="Qwen"
  fi
  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/DPIC/run_dpic.py \
  --task rebuttal \
  --dataset $model \
  --mode test \
  --dpic_ckpt ./detectors/DPIC/weights/v1/rebuttal/dpic_${base_model}_ep4_thres0.0_n2000.pth \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample
done