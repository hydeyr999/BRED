datasets=("create" "rewrite" "summary" "refine" "expand" "translate")
frac="0.2"
device='cuda:2'
DATANUM=2000
EPOCHS=4
LR=1e-4
BETA=0.05

# cross-model
#for dataset in "${datasets[@]}"; do
#  op2="polish"
#  echo "Running with model: $dataset"
#  python detectors/imbd/run_spo.py \
#    --datanum=$DATANUM \
#    --task_name op-co \
#    --epochs=$EPOCHS \
#    --lr=$LR \
#    --beta=$BETA \
#    --device=$device \
#    --train_dataset "$dataset" \
#    --eval_dataset "$dataset" \
#    --frac=$frac \
#    --op2=$op2
#done

## eval
for dataset in ${datasets[@]}; do
  if [[ "$dataset" == "create" || "$dataset" == "translate" ]]; then
    op2s=("rewrite" "polish" "refine" "expand")
  elif [[ "$dataset" == "expand" || "$dataset" == "refine" || "$dataset" == "summary" ]]; then
    op2s=("rewrite" "polish")
  elif [[ "$dataset" == "rewrite" ]]; then
    op2s=("polish" "refine" "expand")
  fi
  for op2 in "${op2s[@]}"; do
    echo "Processing cross-domain: $dataset, op2: $op2,flag 1"
    python detectors/imbd/run_spo.py \
      --eval_only True \
      --from_pretrained "./detectors/imbd/ckpt/v1/op-co/${dataset}_n2000_ep4_spo_lr_0.0001_beta_0.05_a_1" \
      --train_dataset "$dataset" \
      --eval_dataset "$dataset" \
      --device $device \
      --task_name op-co \
      --frac $frac \
      --op2 $op2 \
      --flag "1"
  done

    for op2 in "${op2s[@]}"; do
    echo "Processing cross-domain: $dataset, op2: $op2,flag 2"
    python detectors/imbd/run_spo.py \
      --eval_only True \
      --from_pretrained "./detectors/imbd/ckpt/v1/op-co/${dataset}_n2000_ep4_spo_lr_0.0001_beta_0.05_a_1" \
      --train_dataset "$dataset" \
      --eval_dataset "$dataset" \
      --device $device \
      --task_name op-co \
      --frac $frac \
      --op2 $op2 \
      --flag "2"
  done
done