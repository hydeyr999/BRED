domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
n_sample=2000
epochs=4
threshold="0.0"
device="cuda:0"
multilens=("10" "50" "100" "200" "500")

for multilen in ${multilens[@]}; do
    for domain in ${domains[@]}; do
      echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
      python detectors/opensource/mpu.py \
      --task cross-domain \
      --dataset $domain \
      --device $device \
      --mpu_model ./detectors/mpu/results/v1/cross-domain/mpu_${domain}_ep$epochs\_thres$threshold\_n$n_sample/complete-${epochs} \
      --multilen $multilen
    done
done