operations=("create" "rewrite" "summary" "refine" "expand" "translate")
models=("llama" "deepseek" "gpt4o" "Qwen")
models_2=("gpt_5" "gpt_5_thinking" "deepseek_thinking" "Qwen_thinking")
device='cuda:3'

## Generate data for task sup1
#for model in "${models_2[@]}"; do
#  echo "Generating data for task sup1 on $model"
#  python detectors/ling-based/gltrppl.py \
#  --task thinking \
#  --dataset $model \
#  --run_feats True \
#  --save_feats True \
#  --mode test \
#  --device $device
#done

## Generate data for task sup2 and sup3.sh
for operation in "${operations[@]}"; do
  echo "Generating data for task sup2 on $operation"
  python detectors/ling-based/gltrppl.py \
  --task op-co \
  --dataset $operation \
  --run_feats True \
  --save_feats True \
  --mode train \
  --device $device
done

for operation in "${operations[@]}"; do
  echo "Generating data for task sup2 on $operation"
  python detectors/ling-based/gltrppl.py \
  --task op-co \
  --dataset $operation \
  --run_feats True \
  --save_feats True \
  --mode test \
  --device $device
done

for model in "${models[@]}"; do
  echo "Generating data for task sup3 on $model"
  python detectors/ling-based/gltrppl.py \
  --task llm-co \
  --dataset $model \
  --run_feats True \
  --save_feats True \
  --mode train \
  --device $device
done

for model in "${models[@]}"; do
  echo "Generating data for task sup3 on $model"
  python detectors/ling-based/gltrppl.py \
  --task llm-co \
  --dataset $model \
  --run_feats True \
  --save_feats True \
  --mode test \
  --device $device
done