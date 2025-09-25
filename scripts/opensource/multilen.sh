domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
multilens=("10" "50" "100" "200" "500")
device="cuda:1"

for multilen in ${multilens[@]}; do
  echo "running $multilen"
  for domain in ${domains[@]}; do
    echo "running $domain on radar"
    python detectors/opensource/radar.py --task cross-domain --dataset $domain --device $device --multilen $multilen
    echo "running $domain on mpu"
    python detectors/opensource/mpu.py --task cross-domain --dataset $domain --device $device --multilen $multilen
    echo "running $domain on openai"
    python detectors/opensource/openai.py --task cross-domain --dataset $domain --device $device --multilen $multilen
    echo "running $domain on argugpt"
    python detectors/opensource/argugpt.py --task cross-domain --dataset $domain --device $device --multilen $multilen
    echo "running $domain on roberta-hc3"
    python detectors/opensource/roberta-HC3.py --task cross-domain --dataset $domain --device $device --multilen $multilen
  done
done