models=("gpt_5" "gpt_5_thinking" "deepseek_thinking" "Qwen_thinking")
multilen=0
device='cuda:2'

for model in ${models[@]}; do
  echo "Evaluating model: $model"
  python detectors/detectgpt/fast_detect_gpt.py \
  --task thinking \
  --dataset $model \
  --device $device \
  --multilen $multilen
done