domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
#operations=("create" "rewrite" "summary" "refine" "polish" "translate")
operations=("polish")
epochs=4
n_sample=2000
multilen=0
device='cuda:1'

#for domain in ${domains[@]}; do
#  echo "Testing domain: $domain, epochs: $epochs, n_sample: $n_sample"
#  python detectors/DPIC/run_dpic.py \
#  --task cross-domain \
#  --dataset $domain \
#  --mode test \
#  --dpic_ckpt ./detectors/DPIC/weights/v1/cross-domain/n${n_sample}/dpic_$domain\_ep$epochs\_thres0_n$n_sample.pth \
#  --device $device \
#  --epochs $epochs \
#  --n_sample $n_sample \
#  --multilen $multilen
#done

#for model in ${models[@]}; do
#  echo "Testing model: $model, epochs: $epochs, n_sample: $n_sample"
#  python detectors/DPIC/run_dpic.py --task cross-model --dataset $domain --mode test \
#  --dpic_ckpt ./detectors/DPIC/weights/cross-model/dpic_$model\_ep$epochs\_n$n_sample.pth \
#  --device $device --epochs $epochs --n_sample $n_sample --multilen $multilen
#done

for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, n_sample: $n_sample"
  python detectors/DPIC/run_dpic.py \
  --task cross-operation \
  --dataset $operation \
  --mode test \
  --dpic_ckpt ./detectors/DPIC/weights/v1/cross-operation/n${n_sample}/dpic_$operation\_ep$epochs\_thres0_n$n_sample.pth \
  --device $device \
  --epochs $epochs \
  --n_sample $n_sample \
  --multilen $multilen
done