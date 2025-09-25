domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
multilens=("10" "50" "100" "200" "500")
epochs=4
n_sample=2000
device='cuda:2'

for multilen in "${multilens[@]}"; do
  echo "Testing multilens: $multilen"
  for domain in ${domains[@]}; do
    echo "Testing domain: $domain, epochs: $epochs, n_sample: $n_sample"
    python detectors/DPIC/run_dpic.py \
    --task cross-domain \
    --dataset $domain \
    --mode test \
    --dpic_ckpt ./detectors/DPIC/weights/v1/cross-domain/n${n_sample}/dpic_$domain\_ep$epochs\_thres0_n$n_sample.pth \
    --device $device \
    --epochs $epochs \
    --n_sample $n_sample \
    --multilen $multilen
  done
done