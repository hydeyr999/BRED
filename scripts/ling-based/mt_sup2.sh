datasets=("create" "rewrite" "summary" "refine" "expand" "translate")
frac="0.2"
funcs=("likelihood" "rank" "logrank" "entropy")

#for func in ${funcs[@]}; do
#  for dataset in ${datasets[@]}; do
#    echo "Training on $dataset, function $func"
#    python detectors/ling-based/metrics.py \
#    --task op-co \
#    --dataset $dataset \
#    --mode train \
#    --func $func \
#    --frac $frac
#  done
#done

## Evaluate
for func in ${funcs[@]}; do
  for dataset in ${datasets[@]}; do
    if [[ "$dataset" == "create" || "$dataset" == "translate" ]]; then
      op2s=("rewrite" "polish" "refine" "expand")
    elif [[ "$dataset" == "expand" || "$dataset" == "refine" || "$dataset" == "summary" ]]; then
      op2s=("rewrite" "polish")
    elif [[ "$dataset" == "rewrite" ]]; then
      op2s=("polish" "refine" "expand")
    fi
    for op2 in ${op2s[@]}; do
      echo "Evaluating on $dataset, $op2, func=$func, flag=1"
      python detectors/ling-based/metrics.py \
      --task op-co \
      --dataset $dataset \
      --classifier ./detectors/ling-based/classifier/v1/op-co/${func}_$dataset\_n${frac}.joblib \
      --mode test \
      --frac $frac \
      --op2 $op2 \
      --func $func \
      --flag "1"
    done
    for op2 in ${op2s[@]}; do
      echo "Evaluating on $dataset, $op2, func=$func, flag=2"
      python detectors/ling-based/metrics.py \
      --task op-co \
      --dataset $dataset \
      --classifier ./detectors/ling-based/classifier/v1/op-co/${func}_$dataset\_n${frac}.joblib \
      --mode test \
      --frac $frac \
      --op2 $op2 \
      --func $func \
      --flag "2"
    done
  done
done