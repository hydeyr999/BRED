domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
epochs=10
#threshold="0.0"
thresholds=("0.01" "0.001")
n_sample=2000
#n_samples=(250 500 1000)
multilen=0
device='cuda:1'

for threshold in ${thresholds[@]}; do
#for n_sample in ${n_samples[@]}; do
for domain in ${domains[@]}; do
  echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/deberta/run_roberta.py \
  --task cross-domain \
  --dataset $domain \
  --mode test \
  --roberta_model ./detectors/deberta/weights/v1/cross-domain/thres${threshold}/roberta_$domain\_ep$epochs\_thres$threshold\_n$n_sample \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample \
  --multilen $multilen
done

for model in ${models[@]}; do
  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/deberta/run_roberta.py \
  --task cross-model \
  --dataset $model \
  --mode test \
  --roberta_model ./detectors/deberta/weights/v1/cross-model/thres${threshold}/roberta_$model\_ep$epochs\_thres$threshold\_n$n_sample \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample \
  --multilen $multilen
done

for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/deberta/run_roberta.py \
  --task cross-operation \
  --dataset $operation \
  --mode test \
  --roberta_model ./detectors/deberta/weights/v1/cross-operation/thres${threshold}/roberta_$operation\_ep$epochs\_thres$threshold\_n$n_sample \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample \
  --multilen $multilen
done
#done
done