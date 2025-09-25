domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
languages=("ch" "de" "fr")
device="cuda:1"

for domain in ${domains[@]}; do
  echo "running $domain on radar"
  python detectors/opensource/radar.py --task cross-domain --dataset $domain --device $device
  echo "running $domain on mpu"
  python detectors/opensource/mpu.py --task cross-domain --dataset $domain --device $device
  echo "running $domain on openai"
  python detectors/opensource/openai.py --task cross-domain --dataset $domain --device $device
  echo "running $domain on argugpt"
  python detectors/opensource/argugpt.py --task cross-domain --dataset $domain --device $device
  echo "running $domain on roberta-hc3"
  python detectors/opensource/roberta-HC3.py --task cross-domain --dataset $domain --device $device
done

for model in ${models[@]}; do
  echo "running $model on radar"
  python detectors/opensource/radar.py --task cross-model --dataset $model --device $device
  echo "running $model on mpu"
  python detectors/opensource/mpu.py --task cross-model --dataset $model --device $device
  echo "running $model on openai"
  python detectors/opensource/openai.py --task cross-model --dataset $model --device $device
  echo "running $model on argugpt"
  python detectors/opensource/argugpt.py --task cross-model --dataset $model --device $device
  echo "running $model on roberta-hc3"
  python detectors/opensource/roberta-HC3.py --task cross-model --dataset $model --device $device
done

for operation in ${operations[@]}; do
  echo "running $operation on radar"
  python detectors/opensource/radar.py --task cross-operation --dataset $operation --device $device
  echo "running $operation on mpu"
  python detectors/opensource/mpu.py --task cross-operation --dataset $operation --device $device
  echo "running $operation on openai"
  python detectors/opensource/openai.py --task cross-operation --dataset $operation --device $device
  echo "running $operation on argugpt"
  python detectors/opensource/argugpt.py --task cross-operation --dataset $operation --device $device
  echo "running $operation on roberta-hc3"
  python detectors/opensource/roberta-HC3.py --task cross-operation --dataset $operation --device $device
done