domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "translate" "expand")
epochs=2
threshold="0.0"
n_samples=("2000")
device='cuda:3'

for n_sample in ${n_samples[@]}; do
for domain in ${domains[@]}; do
  echo "Training on $domain, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
  python detectors/llama/run_llama.py \
  --task cross-domain \
  --dataset $domain \
  --mode train \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample
done

#for model in ${models[@]}; do
#  echo "Training on $model, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
#  python detectors/llama/run_llama.py --task cross-model --dataset $model --mode train \
#  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample
#done

for operation in ${operations[@]}; do
  echo "Training on $operation, epochs=$epochs, threshold=$threshold, n_sample=$n_sample"
  python detectors/llama/run_llama.py \
  --task cross-operation \
  --dataset $operation \
  --mode train \
  --device $device \
  --epochs $epochs \
  --threshold $threshold \
  --n_sample $n_sample
done
done