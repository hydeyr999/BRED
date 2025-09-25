domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
#operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
operations=('translate')
epochs=20
threshold='0.1'
n_sample=250
multilen=0
device='cuda:1'

python detectors/DPIC/run_dpic.py --mode train --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample \
--imbddata True --dataset imbd --translate True

#for domain in ${domains[@]}; do
#  echo "Testing domain: $domain, epochs: $epochs, n_sample: $n_sample"
#  python detectors/DPIC/run_dpic.py --task cross-domain --dataset $domain --mode test \
#  --dpic_ckpt ./detectors/DPIC/weights/otherdata/dpic_imbd_ep$epochs\_thres$threshold\_n$n_sample.pth \
#  --device $device --epochs $epochs --n_sample $n_sample --multilen $multilen --thres $threshold
#done
#
#for model in ${models[@]}; do
#  echo "Testing model: $model, epochs: $epochs, n_sample: $n_sample"
#  python detectors/DPIC/run_dpic.py --task cross-model --dataset $model --mode test \
#  --dpic_ckpt ./detectors/DPIC/weights/otherdata/dpic_imbd_ep$epochs\_thres$threshold\_n$n_sample.pth \
#  --device $device --epochs $epochs --n_sample $n_sample --multilen $multilen --thres $threshold
#done

for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, n_sample: $n_sample"
  python detectors/DPIC/run_dpic.py --task cross-operation --dataset $operation --mode test \
  --dpic_ckpt ./detectors/DPIC/weights/otherdata/dpic_imbd_ep$epochs\_thres$threshold\_n$n_sample\_True.pth \
  --device $device --epochs $epochs --n_sample $n_sample --multilen $multilen --thres $threshold
done