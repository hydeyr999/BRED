domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
epochs=4
threshold="0.0"
n_sample=2000
multilen=0
device='cuda:2'

#for threshold in ${thresholds[@]}; do
for domain in ${domains[@]}; do
  echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/deberta/run_deberta.py \
  --task cross-domain \
  --dataset $domain \
  --mode test \
  --deberta_model ./detectors/deberta/weights/v1/cross-domain/n${n_sample}/deberta_$domain\_ep$epochs\_thres$threshold\_n$n_sample \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample \
  --multilen $multilen
done

#for model in ${models[@]}; do
#  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
#  python detectors/deberta/run_deberta.py --task cross-model --dataset $model --mode test \
#  --deberta_model ./detectors/deberta/weights/cross-model/deberta_$model\_ep$epochs\_thres$threshold\_n$n_sample \
#  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen
#done

for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/deberta/run_deberta.py \
  --task cross-operation \
  --dataset $operation \
  --mode test \
  --deberta_model ./detectors/deberta/weights/v1/cross-operation/n${n_sample}/deberta_$operation\_ep$epochs\_thres$threshold\_n$n_sample \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample \
  --multilen $multilen
done
#done
