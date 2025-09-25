domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
#operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
operations=('translate')
languages=("ch" "de" "fr")
n_sample=250
multilen=0
device="cuda:0"

#python detectors/ling-based/gltrppl.py --imbddata True --translate True --dataset HC3 --n_sample $n_sample --device $device --mode train --func ppl

#for domain in ${domains[@]}; do
#  echo "Testing on $domain, n_sample=$n_sample"
#  python detectors/ling-based/gltrppl.py --task cross-domain --dataset $domain \
#  --mode test --classifier ./detectors/ling-based/classifier/otherdata/ppl_imbd_n${n_sample}.pkl \
#  --n_sample $n_sample --multilen $multilen --device $device --imbddata True --func ppl
#done
#
#for model in ${models[@]}; do
#  echo "Testing on $model, n_sample=$n_sample"
#  python detectors/ling-based/gltrppl.py --task cross-model --dataset $model \
#  --mode test --classifier ./detectors/ling-based/classifier/otherdata/ppl_imbd_n${n_sample}.pkl \
#  --n_sample $n_sample --multilen $multilen --device $device --imbddata True --func ppl
#done

#for operation in ${operations[@]}; do
#  echo "Testing on $operation, n_sample=$n_sample"
#  python detectors/ling-based/gltrppl.py --task cross-operation --dataset $operation \
#  --mode test --classifier ./detectors/ling-based/classifier/otherdata/ppl_imbd_n${n_sample}\_True.pkl \
#  --n_sample $n_sample --multilen $multilen --device $device --imbddata True --func ppl --translate True
#done

#for domain in ${domains[@]}; do
#  echo "Testing on $operation, n_sample=$n_sample, subdomain=$domain"
#  python detectors/ling-based/gltrppl.py --task cross-operation --dataset translate \
#  --mode test --classifier ./detectors/ling-based/classifier/otherdata/ppl_imbd_n${n_sample}.pkl \
#  --n_sample $n_sample --multilen $multilen --device $device --imbddata True --func ppl --subdomain $domain
#done

for language in ${languages[@]}; do
#for domain in ${domains[@]}; do
  echo "Testing on translate, n_sample=$n_sample, subdomain=$domain, language=$language"
  python detectors/ling-based/gltrppl.py --task cross-operation --dataset translate \
  --mode test --classifier ./detectors/ling-based/classifier/otherdata/ppl_HC3_n${n_sample}\_True.pkl \
  --n_sample $n_sample --multilen $multilen --device $device --imbddata True --func ppl --translate True \
  --language $language #--subdomain $domain
#done
done