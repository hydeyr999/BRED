models=("llama" "deepseek" "gpt4o" "Qwen")
models_2=('llama' 'llama_8b_instruct' 'deepseek' 'gpt4o' 'gpt4o_large' 'Qwen' 'Qwen_7b' 'Qwen_8b')
epochs=4
threshold="0.0"
n_sample=2000
device="cuda:1"

#for model in ${models[@]}; do
#  echo "Training on $model, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
#  python detectors/deberta/run_distilbert.py \
#  --task cross-model \
#  --dataset $model \
#  --mode train \
#  --device $device \
#  --epochs $epochs \
#  --threshold $threshold \
#  --n_sample $n_sample \
#  --re True
#done

## Evaluate detectors
for model in ${models_2[@]}; do
  echo "Evaluating on $model, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
  if [[ "$model" == "llama" || "$model" == "llama_8b_instruct" ]]; then
        base_model="llama"
    elif [[ "$model" == "deepseek" ]]; then
        base_model="deepseek"
    elif [[ "$model" == "gpt4o" || "$model" == "gpt4o_large" ]]; then
        base_model="gpt4o"
    elif [[ "$model" == "Qwen" || "$model" == "Qwen_7b" || "$model" == "Qwen_8b" ]]; then
        base_model="Qwen"
    fi
  python detectors/deberta/run_distilbert.py \
  --task cross-model \
  --dataset $model \
  --mode test \
  --distilbert_model ./detectors/deberta/weights/v1/cross-model/re/distilbert_$base_model\_ep$epochs\_thres$threshold\_n$n_sample \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample \
  --re True
done