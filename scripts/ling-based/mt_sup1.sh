models=("llama" "deepseek" "gpt" "Qwen")
models_2=("llama" "gpt_4o" "gpt_5" "gpt_5_thinking" "Qwen" "Qwen_thinking" "deepseek" "deepseek_thinking")
n_sample=2000
device='cuda:0'
funcs=("likelihood" "rank" "logrank" "entropy")

#for func in ${funcs[@]}; do
#  for model in ${models[@]}; do
#    echo "Training on $model, function $func"
#    python detectors/ling-based/metrics.py \
#    --task thinking \
#    --dataset $model \
#    --mode train \
#    --func $func \
#    --n_sample $n_sample \
#    --device $device
#  done
#done

## Evaluate
for func in ${funcs[@]}; do
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
    echo "Processing model: $model"
    python detectors/ling-based/metrics.py \
    --task thinking \
    --dataset $model \
    --classifier ./detectors/ling-based/classifier/v1/thinking/${func}_${base_model}_n${n_sample}_multilen0.joblib \
    --device $device \
    --n_sample $n_sample \
    --func $func
  done
done
