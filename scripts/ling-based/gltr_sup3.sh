datasets=("llama" "deepseek" "gpt4o" "Qwen")
frac="0.2"
funcs=('gltr' 'ppl')

#for func in ${funcs[@]}; do
#  for dataset in ${datasets[@]}; do
#    echo "Training on $dataset, func=$func"
#    python detectors/ling-based/gltrppl.py \
#    --task llm-co \
#    --dataset $dataset \
#    --device $device \
#    --mode train \
#    --frac $frac \
#    --func $func
#  done
#done

## Evaluate
for func in ${funcs[@]}; do
  for dataset in ${datasets[@]}; do
    if [[ "$dataset" == "llama" ]]; then
      model2s=("deepseek" "gpt4o" "Qwen")
    elif [[ "$dataset" == "deepseek" ]]; then
      model2s=("llama" "gpt4o" "Qwen")
    elif [[ "$dataset" == "gpt4o" ]]; then
      model2s=("llama" "deepseek" "Qwen")
    elif [[ "$dataset" == "Qwen" ]]; then
      model2s=("llama" "deepseek" "gpt4o")
    fi
    for model2 in ${model2s[@]}; do
      echo "Evaluating on $dataset, $model2, func=$func, flag=1"
      python detectors/ling-based/gltrppl.py \
      --task llm-co \
      --dataset $dataset \
      --classifier ./detectors/ling-based/classifier/v1/llm-co/${func}_$dataset\_n${frac}.pkl \
      --mode test \
      --frac $frac \
      --model2 $model2 \
      --func $func \
      --flag "1"
    done
    for model2 in ${model2s[@]}; do
      echo "Evaluating on $dataset, $model2, func=$func, flag=2"
      python detectors/ling-based/gltrppl.py \
      --task llm-co \
      --dataset $dataset \
      --classifier ./detectors/ling-based/classifier/v1/llm-co/${func}_$dataset\_n${frac}.pkl \
      --mode test \
      --frac $frac \
      --model2 $model2 \
      --func $func \
      --flag "2"
    done
  done
done
