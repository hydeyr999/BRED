domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
languages=("ch" "de" "fr")
multilen=0
device='cuda:3'

#for domain in ${domains[@]}; do
#  echo "Evaluating domain: $domain"
#  python detectors/baselines/baselines.py --task cross-domain --dataset $domain --device $device --multilen $multilen
#done

#for model in ${models[@]}; do
#  echo "Evaluating model: $model"
#  python detectors/baselines/baselines.py --task cross-model --dataset $model --device $device --multilen $multilen
#done

#for operation in ${operations[@]}; do
#  echo "Evaluating operation: $operation"
#  python detectors/baselines/baselines.py --task cross-operation --dataset $operation --device $device --multilen $multilen
#done

for language in ${languages[@]}; do
  echo "Evaluating operation: translate, language: $language"
  python detectors/baselines/baselines.py --task cross-operation --dataset translate --device $device \
  --language $language --multilen $multilen
done
