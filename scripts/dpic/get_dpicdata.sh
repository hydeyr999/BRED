domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "ds" "4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
rebuttals=("Qwen_8b")
device="cuda:0"

#for domain in "${domains[@]}"; do
#  for model in "${models[@]}"; do
#    for operation in "${operations[@]}"; do
#      echo "Running DPIC for $operation on $domain with $model"
#      python detectors/DPIC/run_dpic.py \
#      --run_generate True \
#      --ori_data_path ./data_gen/LLM-texts-new/$operation/$domain\_$model\_$operation.csv \
#      --gen_save_path ./data_gen/LLM-texts-new/$operation/$domain\_$model\_$operation\_dpic.csv \
#      --device $device
#    done
#  done
#done

for item in "${rebuttals[@]}"; do
  echo "Generating dpic data for rebuttal dataset: $item"
  python detectors/DPIC/run_dpic.py \
    --run_generate True \
    --task rebuttal \
    --dataset $item \
    --gen_save_path ./detectors/DPIC/dpic_data/v1/rebuttal/${item}_sample.json \
    --device $device
done