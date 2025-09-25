domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
#operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
operations=('translate')
languages=("de" "fr")
epochs=10
threshold="0.01"
n_sample=250
multilen=0
device='cuda:2'

#python detectors/llama/run_llama.py --dataset HC3 --mode train --device $device \
#--epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata True --translate True

#for domain in ${domains[@]}; do
#  echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
#  python detectors/llama/run_llama.py --task cross-domain --dataset $domain --mode test \
#  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata $imbddata
#done
#
#for model in ${models[@]}; do
#  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
#  python detectors/llama/run_llama.py --task cross-model --dataset $model --mode test \
#  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata $imbddata
#done

#for operation in ${operations[@]}; do
#  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
#  python detectors/llama/run_llama.py --task cross-operation --dataset $operation --mode test \
#  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata True
#done

for language in ${languages[@]}; do
for operation in ${operations[@]}; do
  echo "Testing operation translate, epochs: $epochs, threshold: $threshold, n_sample: $n_sample, language: $language"
  python detectors/llama/run_llama.py --task cross-operation --dataset $operation --mode test \
  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen \
  --imbddata True --translate True --language $language
done
done