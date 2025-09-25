datasets=("create" "rewrite" "summary" "refine" "expand" "translate")
frac="0.2"
device='cuda:3'

#for dataset in ${datasets[@]}; do
#  echo "Training on $dataset"
#  python detectors/ling-based/ghostbuster.py \
#  --task op-co \
#  --dataset $dataset \
#  --mode train \
#  --frac $frac
#done

## Eval
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
    python detectors/ling-based/ghostbuster.py \
    --task op-co \
    --dataset $dataset \
    --classifier ./detectors/ling-based/classifier/v1/op-co/gb_$dataset\_n${frac}.pkl \
    --mode test \
    --frac $frac \
    --op2 $op2 \
    --flag "1"
  done

  for op2 in ${op2s[@]}; do
    echo "Evaluating on $dataset, $op2, func=$func, flag=2"
    python detectors/ling-based/ghostbuster.py \
    --task op-co \
    --dataset $dataset \
    --classifier ./detectors/ling-based/classifier/v1/op-co/gb_$dataset\_n${frac}.pkl \
    --mode test \
    --frac $frac \
    --op2 $op2 \
    --flag "2"
  done
done