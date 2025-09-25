domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "expand" "translate")
n_sample=2000
#n_samples=(250 500 1000)
multilen=0
device="cuda:0"
funcs=("likelihood" "rank" "logrank" "entropy")

for n_sample in ${n_samples[@]}; do
for func in ${funcs[@]}; do
  for domain in ${domains[@]}; do
    echo "Processing domain: $domain"
    python detectors/ling-based/metrics.py \
    --task cross-domain \
    --dataset $domain \
    --classifier ./detectors/ling-based/classifier/v1/cross-domain/n${n_sample}/${func}_${domain}_n${n_sample}_multilen${multilen}.joblib \
    --device $device \
    --n_sample $n_sample \
    --multilen $multilen \
    --func $func
  done

  for model in ${models[@]}; do
    echo "Processing model: $model"
    python detectors/ling-based/metrics.py \
    --task cross-model \
    --dataset $model \
    --classifier ./detectors/ling-based/classifier/v1/cross-model/n${n_sample}/${func}_${model}_n${n_sample}_multilen${multilen}.joblib \
    --device $device \
    --n_sample $n_sample \
    --multilen $multilen \
    --func $func
  done

  for operation in ${operations[@]}; do
    echo "Processing operation: $operation"
    python detectors/ling-based/metrics.py \
    --task cross-operation \
    --dataset $operation \
    --classifier ./detectors/ling-based/classifier/v1/cross-operation/n${n_sample}/${func}_${operation}_n${n_sample}_multilen${multilen}.joblib \
    --device $device \
    --n_sample $n_sample \
    --multilen $multilen \
    --func $func
  done
done
done

