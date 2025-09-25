domains=("xsum" "writingprompts" "pubmedqa" "squad" "openreview" "blog" "tweets")
#models=("llama" "deepseek" "gpt4o" "Qwen")
models=("llama_8b_instruct" "gpt4o_large" "Qwen_7b" "Qwen_8b")
operations=("create" "rewrite" "summary" "refine" "polish" "expand" "translate")

for model in "${models[@]}"; do
  echo "Processing model: $model"
  python detectors/ling-based/metrics.py \
    --task rebuttal \
    --dataset $model \
    --run_metrics True\
    --save_metrics True
done