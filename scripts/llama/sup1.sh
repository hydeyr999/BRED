models=("llama" "deepseek" "gpt" "Qwen")
models_2=("llama" "gpt_4o" "gpt_5" "gpt_5_thinking" "Qwen" "Qwen_thinking" "deepseek" "deepseek_thinking")
epochs=4
threshold="0.0"
n_sample=2000
device='cuda:2'

#for model in ${models[@]}; do
#  echo "Training on $model, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
#  python detectors/llama/run_llama.py \
#  --task thinking \
#  --dataset $model \
#  --mode train \
#  --device $device \
#  --epochs $epochs \
#  --threshold $threshold \
#  --n_sample $n_sample
#done

## Eval

for model in ${models_2[@]}; do
  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/llama/run_llama.py \
  --task thinking \
  --dataset $model \
  --mode test \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample
done