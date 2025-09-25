#domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
domains=("blog" "tweets")
#multilens=("10" "50" "100" "200" "500")
multilens=("500")
n_sample=2000
device="cuda:0"
funcs=("likelihood" "rank" "logrank" "entropy")

for multilen in ${multilens[@]}; do
  for func in ${funcs[@]}; do
    for domain in ${domains[@]}; do
      echo "Processing domain: $domain"
      python detectors/ling-based/metrics.py \
      --task cross-domain \
      --dataset $domain \
      --classifier ./detectors/ling-based/classifier/v1/cross-domain/n${n_sample}/${func}_${domain}_n${n_sample}_multilen0.joblib \
      --device $device \
      --n_sample $n_sample \
      --multilen $multilen \
      --func $func
    done
  done
done