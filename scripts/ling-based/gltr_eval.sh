domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "refine" "polish" "expand" "translate")
multilen=0
n_sample=2000
device="cuda:0"
#funcs=('gltr' 'ppl')
funcs=('ppl')

for func in ${funcs[@]}; do
for domain in ${domains[@]}; do
  echo "Testing on $domain, n_sample=$n_sample"
  python detectors/ling-based/gltrppl.py \
  --task cross-domain \
  --dataset $domain \
  --classifier ./detectors/ling-based/classifier/v1/cross-domain/n${n_sample}/${func}_$domain\_n${n_sample}.pkl \
  --multilen $multilen \
  --n_sample $n_sample \
  --device $device \
  --mode test \
  --func $func
done

#for model in ${models[@]}; do
#  echo "Testing on $model, n_sample=$n_sample"
#  python detectors/ling-based/gltrppl.py --task cross-model --dataset $model \
#  --classifier ./detectors/ling-based/classifier/cross-model/gltr_$model\_n${n_sample}.pkl --multilen $multilen \
#  --n_sample $n_sample --device $device --mode test --func $func
#done

for operation in ${operations[@]}; do
  echo "Testing on $operation, n_sample=$n_sample"
  python detectors/ling-based/gltrppl.py \
  --task cross-operation \
  --dataset $operation \
  --classifier ./detectors/ling-based/classifier/v1/cross-operation/n${n_sample}/${func}_$operation\_n${n_sample}.pkl \
  --multilen $multilen \
  --n_sample $n_sample \
  --device $device \
  --mode test \
  --func $func
done
done