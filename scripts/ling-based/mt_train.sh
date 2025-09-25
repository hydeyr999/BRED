domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "expand" "translate")
n_sample=1000
device="cuda:0"
funcs=("likelihood" "rank" "logrank" "entropy")

for func in ${funcs[@]}; do
  for domain in ${domains[@]}; do
    echo "Training on $domain, function $func"
    python detectors/ling-based/metrics.py \
    --task cross-domain \
    --dataset $domain \
    --mode train \
    --func $func \
    --n_sample $n_sample \
    --device $device
  done

  for model in ${models[@]}; do
    echo "Training on $model, function $func"
    python detectors/ling-based/metrics.py \
    --task cross-model \
    --dataset $model \
    --mode train \
    --func $func \
    --n_sample $n_sample \
    --device $device
  done

  for operation in ${operations[@]}; do
    echo "Training on $operation, function $func"
    python detectors/ling-based/metrics.py \
    --task cross-operation \
    --dataset $operation \
    --mode train \
    --func $func \
    --n_sample $n_sample \
    --device $device
  done
done


