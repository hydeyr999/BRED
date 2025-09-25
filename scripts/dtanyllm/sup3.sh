datasets=("llama" "deepseek" "gpt4o" "Qwen")
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
#      --task llm-co \
#      --train_data_path ./data/v1/llm-co/train/${dataset}_sample_train_n${frac}.json \
#      --train_data_format detect \
#      --save_directory ./detectors/detectanyllm/ckpt/v1/llm-co/ \
#      --wandb_dir ./detectors/detectanyllm/log/v1/llm-co/ \
#      --train_method ${TRAIN_METHOD} \
#      --num_epochs ${num_epochs} \
#      --frac ${frac}
#done

## eval
for dataset in "${datasets[@]}"; do
  if [[ "$dataset" == "llama" ]]; then
    model2s=("deepseek" "gpt4o" "Qwen")
  elif [[ "$dataset" == "deepseek" ]]; then
    model2s=("llama" "gpt4o" "Qwen")
  elif [[ "$dataset" == "gpt4o" ]]; then
    model2s=("llama" "deepseek" "Qwen")
  elif [[ "$dataset" == "Qwen" ]]; then
    model2s=("llama" "deepseek" "gpt4o")
  fi
  for model2 in "${model2s[@]}"; do
    echo "Evaluating ${dataset} -> ${model2}, flag 1"
    accelerate launch detectors/detectanyllm/eval.py \
        --task llm-co \
        --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/llm-co/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${dataset}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
        --eval_data_path ./data/v1/llm-co/test/${dataset}_sample_test_n${frac}.json \
        --eval_data_format detect \
        --wandb True \
        --train_method ${TRAIN_METHOD} \
        --eval_batch_size 4 \
        --frac ${frac} \
        --model2 ${model2} \
        --flag "1" \
        --save_dir ./detectors/detectanyllm/results/v1/llm-co \
        --save_file ${dataset}_${model2}_n${frac}_1.json
  done

  for model2 in "${model2s[@]}"; do
    echo "Evaluating ${dataset} -> ${model2}, flag 2"
    accelerate launch detectors/detectanyllm/eval.py \
        --task llm-co \
        --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/llm-co/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${dataset}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
        --eval_data_path ./data/v1/llm-co/test/${dataset}_sample_test_n${frac}.json \
        --eval_data_format detect \
        --wandb True \
        --train_method ${TRAIN_METHOD} \
        --eval_batch_size 4 \
        --frac ${frac} \
        --model2 ${model2} \
        --flag "2" \
        --save_dir ./detectors/detectanyllm/results/v1/llm-co \
        --save_file ${dataset}_${model2}_n${frac}_2.json
  done
done