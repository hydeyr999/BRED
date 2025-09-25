datasets=("create" "rewrite" "summary" "refine" "expand" "translate")
frac="0.2"
TRAIN_METHOD="DDL"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="None"
num_epochs=4
n_sample=2000

export WANDB_MODE=offline
#for dataset in ${datasets[@]}; do
#  accelerate launch detectors/detectanyllm/train.py \
#      --scoring_model_name ${SCORING_MODEL} \
#      --reference_model_name ${REFERENCE_MODEL} \
#      --wandb True \
#      --task op-co \
#      --train_data_path ./data/v1/op-co/train/${dataset}_sample_train_n${frac}.json \
#      --train_data_format detect \
#      --save_directory ./detectors/detectanyllm/ckpt/v1/op-co/ \
#      --wandb_dir ./detectors/detectanyllm/log/v1/op-co/ \
#      --train_method ${TRAIN_METHOD} \
#      --num_epochs ${num_epochs} \
#      --frac ${frac}
#done

## eval
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "create" || "$dataset" == "translate" ]]; then
    op2s=("rewrite" "polish" "refine" "expand")
  elif [[ "$dataset" == "expand" || "$dataset" == "refine" || "$dataset" == "summary" ]]; then
    op2s=("rewrite" "polish")
  elif [[ "$dataset" == "rewrite" ]]; then
    op2s=("polish" "refine" "expand")
  fi
  for op2 in "${op2s[@]}"; do
    echo "Evaluating ${dataset} -> ${op2}, flag 1"
    accelerate launch detectors/detectanyllm/eval.py \
        --task op-co \
        --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/op-co/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${dataset}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
        --eval_data_path ./data/v1/op-co/test/${dataset}_sample_test_n${frac}.json \
        --eval_data_format detect \
        --wandb True \
        --train_method ${TRAIN_METHOD} \
        --eval_batch_size 4 \
        --frac ${frac} \
        --op2 ${op2} \
        --flag "1" \
        --save_dir ./detectors/detectanyllm/results/v1/op-co \
        --save_file ${dataset}_${op2}_n${frac}_1.json
  done

  for op2 in "${op2s[@]}"; do
    echo "Evaluating ${dataset} -> ${op2}, flag 2"
    accelerate launch detectors/detectanyllm/eval.py \
        --task op-co \
        --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/op-co/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${dataset}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
        --eval_data_path ./data/v1/op-co/test/${dataset}_sample_test_n${frac}.json \
        --eval_data_format detect \
        --wandb True \
        --train_method ${TRAIN_METHOD} \
        --eval_batch_size 4 \
        --frac ${frac} \
        --op2 ${op2} \
        --flag "2" \
        --save_dir ./detectors/detectanyllm/results/v1/op-co \
        --save_file ${dataset}_${op2}_n${frac}_2.json
  done
done