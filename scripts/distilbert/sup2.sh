datasets=("create" "rewrite" "summary" "refine" "expand" "translate")
frac="0.2"
epochs=4
threshold="0.0"
device='cuda:1'

## Train
#for dataset in ${datasets[@]}; do
#  echo "Training on $dataset, epochs=$epochs, threshold=$threshold"
#  python detectors/deberta/run_distilbert.py \
#  --task op-co \
#  --dataset $dataset \
#  --mode train \
#  --device $device \
#  --epochs $epochs \
#  --threshold $threshold \
#  --frac $frac
#done

## Evaluate
for dataset in ${datasets[@]}; do
  if [[ "$dataset" == "create" || "$dataset" == "translate" ]]; then
    op2s=("rewrite" "polish" "refine" "expand")
  elif [[ "$dataset" == "expand" || "$dataset" == "refine" || "$dataset" == "summary" ]]; then
    op2s=("rewrite" "polish")
  elif [[ "$dataset" == "rewrite" ]]; then
    op2s=("polish" "refine" "expand")
  fi
  for op2 in ${op2s[@]}; do
    echo "Evaluating on $dataset, epochs=$epochs, threshold=$threshold, op2=$op2"
    python detectors/deberta/run_distilbert.py \
    --task op-co \
    --dataset $dataset \
    --mode test \
    --distilbert_model ./detectors/deberta/weights/v1/op-co/distilbert_$dataset\_ep$epochs\_thres$threshold\_n$frac \
    --device $device \
    --epochs $epochs \
    --threshold $threshold \
    --frac $frac \
    --op2 $op2
  done
done