datasets=("create" "rewrite" "summary" "refine" "expand" "translate")
device="cuda:1"

#for item in "${datasets[@]}"; do
#  echo "Generating dpic data for op-co dataset: $item, train mode"
#  python detectors/DPIC/run_dpic.py \
#    --run_generate True \
#    --task op-co \
#    --dataset $item \
#    --device $device \
#    --mode train
#done

#for item in "${datasets[@]}"; do
#  echo "Generating dpic data for op-co dataset: $item, test mode"
#  python detectors/DPIC/run_dpic.py \
#    --run_generate True \
#    --task op-co \
#    --dataset $item \
#    --device $device
#done

#for item in "${datasets[@]}"; do
#  echo "Generating dpic data for op-co dataset: $item, train mode"
#  python detectors/DPIC/run_dpic.py \
#    --task op-co \
#    --dataset $item \
#    --device $device \
#    --mode train
#done

for item in "${datasets[@]}"; do
  if [[ "$item" == "create" || "$item" == "translate" ]]; then
    op2s=("rewrite" "polish" "refine" "expand")
  elif [[ "$item" == "expand" || "$item" == "refine" || "$item" == "summary" ]]; then
    op2s=("rewrite" "polish")
  elif [[ "$item" == "rewrite" ]]; then
    op2s=("polish" "refine" "expand")
  fi
  for op2 in "${op2s[@]}"; do
    echo "Generating dpic data for op-co dataset: $item, op2: $op2, flag 1"
    python detectors/DPIC/run_dpic.py \
      --task op-co \
      --dataset $item \
      --dpic_ckpt ./detectors/DPIC/weights/v1/op-co/dpic_$item\_ep4_thres0_n250.pth \
      --device $device \
      --op2 $op2 \
      --flag 1
  done

  for op2 in "${op2s[@]}"; do
    echo "Generating dpic data for op-co dataset: $item, op2: $op2, flag 2"
    python detectors/DPIC/run_dpic.py \
      --task op-co \
      --dataset $item \
      --dpic_ckpt ./detectors/DPIC/weights/v1/op-co/dpic_$item\_ep4_thres0_n250.pth \
      --device $device \
      --op2 $op2 \
      --flag 2
  done
done
