models=("llama" "deepseek" "gpt4o" "Qwen")
models_2=('llama' 'llama_8b_instruct' 'deepseek' 'gpt4o' 'gpt4o_large' 'Qwen' 'Qwen_7b' 'Qwen_8b')
n_sample=2000
multilen=0
device="cuda:0"
funcs=("likelihood" "rank" "logrank" "entropy")

#for func in ${funcs[@]}; do
#  for model in ${models[@]}; do
#    echo "Training on $model, function $func"
#    python detectors/ling-based/metrics.py \
#    --task cross-model \
#    --dataset $model \
#    --mode train \
#    --func $func \
#    --n_sample $n_sample \
#    --device $device \
#    --re True
#  done
#done

for func in ${funcs[@]}; do
  for model in ${models_2[@]}; do
    echo "Processing model: $model"
        if [[ "$model" == "llama" || "$model" == "llama_8b_instruct" ]]; then
        base_model="llama"
    elif [[ "$model" == "deepseek" ]]; then
        base_model="deepseek"
    elif [[ "$model" == "gpt4o" || "$model" == "gpt4o_large" ]]; then
        base_model="gpt4o"
    elif [[ "$model" == "Qwen" || "$model" == "Qwen_7b" || "$model" == "Qwen_8b" ]]; then
        base_model="Qwen"
    fi
    python detectors/ling-based/metrics.py \
    --task cross-model \
    --dataset $model \
    --classifier ./detectors/ling-based/classifier/v1/cross-model/re/${func}_${base_model}_n${n_sample}_multilen${multilen}.joblib \
    --device $device \
    --n_sample $n_sample \
    --multilen $multilen \
    --func $func \
    --re True
  done
done

