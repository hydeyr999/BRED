domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
#operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
operations=('translate')
languages=("ch" "de" "fr")
n_sample=250
multilen=0
device="cuda:1"

#getting ghostbuster data
#python detectors/ling-based/ghostbuster.py --imbddata True --dataset HC3 --n_sample $n_sample --device $device --mode train \
#--translate True --run_logprobs True --if_save True

#for language in ${languages[@]}; do
#  echo "Generating logprobs for translate operation, language: $language"
#  python detectors/ling-based/ghostbuster.py --task cross-operation --dataset translate --n_sample $n_sample --device $device --mode test \
#  --translate True --run_logprobs True --if_save True --language $language
#done

#starting from here
#python detectors/ling-based/ghostbuster.py --imbddata True --translate True --dataset HC3 --n_sample $n_sample --device $device --mode train

#for domain in ${domains[@]}; do
#  echo "Testing on $domain, n_sample=$n_sample"
#  python detectors/ling-based/ghostbuster.py --task cross-domain --dataset $domain \
#  --mode test --classifier ./detectors/ling-based/classifier/otherdata/gb_imbd_n${n_sample}.pkl \
#  --n_sample $n_sample --multilen $multilen --device $device --imbddata True
#done
#
#for model in ${models[@]}; do
#  echo "Testing on $model, n_sample=$n_sample"
#  python detectors/ling-based/ghostbuster.py --task cross-model --dataset $model \
#  --mode test --classifier ./detectors/ling-based/classifier/otherdata/gb_imbd_n${n_sample}.pkl \
#  --n_sample $n_sample --multilen $multilen --device $device --imbddata True
#done

#for operation in ${operations[@]}; do
#  echo "Testing on $operation, n_sample=$n_sample"
#  python detectors/ling-based/ghostbuster.py --task cross-operation --dataset $operation \
#  --mode test --classifier ./detectors/ling-based/classifier/otherdata/gb_imbd_n${n_sample}\_True.pkl \
#  --n_sample $n_sample --multilen $multilen --device $device --imbddata True --translate True
#done

#translate start here
#for domain in ${domains[@]}; do
#  echo "Testing on $operation, n_sample=$n_sample, subdomain=$domain"
#  python detectors/ling-based/ghostbuster.py --task cross-operation --dataset translate \
#  --mode test --classifier ./detectors/ling-based/classifier/otherdata/gb_imbd_n${n_sample}.pkl \
#  --n_sample $n_sample --multilen $multilen --device $device --imbddata True --subdomain $domain
#done

for language in ${languages[@]}; do
#for domain in ${domains[@]}; do
  echo "Testing on translate, n_sample=$n_sample, subdomain=$domain, language=$language"
  python detectors/ling-based/ghostbuster.py --task cross-operation --dataset translate \
  --mode test --classifier ./detectors/ling-based/classifier/otherdata/gb_HC3_n${n_sample}\_True.pkl \
  --n_sample $n_sample --multilen $multilen --device $device --imbddata True --translate True \
  --language $language #--subdomain $domain
#done
done