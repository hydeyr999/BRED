datasets=("create" "rewrite" "summary" "refine" "expand" "translate")
frac="0.2"
device='cuda:1'

for dataset in ${datasets[@]}; do
  if [[ "$dataset" == "create" || "$dataset" == "translate" ]]; then
    op2s=("rewrite" "polish" "refine" "expand")
  elif [[ "$dataset" == "expand" || "$dataset" == "refine" || "$dataset" == "summary" ]]; then
    op2s=("rewrite" "polish")
  elif [[ "$dataset" == "rewrite" ]]; then
    op2s=("polish" "refine" "expand")
  fi
  for op2 in ${op2s[@]}; do
    echo "Evaluating dataset: $dataset on detectgpt..."
    python detectors/detectgpt/detectgpt.py \
    --task op-co \
    --dataset $dataset \
    --frac $frac \
    --device $device \
    --op2 $op2
  done
done