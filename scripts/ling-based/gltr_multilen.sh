domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
n_sample=2000
multilens=("10" "50" "100" "200" "500")
device='cuda:2'
funcs=('gltr' 'ppl')

for multilen in ${multilens[@]}; do
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
  done
done
