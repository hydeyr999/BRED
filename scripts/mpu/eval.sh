domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
n_sample=2000
epochs=4
threshold="0.0"
device="cuda:1"

#for n_sample in ${n_samples[@]}; do
for domain in ${domains[@]}; do
  echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/opensource/mpu.py \
  --task cross-domain \
  --dataset $domain \
  --device $device \
  --mpu_model ./detectors/mpu/results/v1/cross-domain/n${n_sample}/mpu_${domain}_ep$epochs\_thres$threshold\_n$n_sample/complete-${epochs} \
  --n_sample $n_sample
done

for model in ${models[@]}; do
  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/opensource/mpu.py \
  --task cross-model \
  --dataset $model \
  --device $device \
  --mpu_model ./detectors/mpu/results/v1/cross-model/n${n_sample}/mpu_${model}_ep$epochs\_thres$threshold\_n$n_sample/complete-${epochs} \
  --n_sample $n_sample
done

for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/opensource/mpu.py \
  --task cross-operation \
  --dataset $operation \
  --device $device \
  --mpu_model ./detectors/mpu/results/v1/cross-operation/n${n_sample}/mpu_${operation}_ep$epochs\_thres$threshold\_n$n_sample/complete-${epochs} \
  --n_sample $n_sample
done
#done