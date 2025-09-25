domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
TRAIN_METHOD="DDL"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="None"
#n_sample=2000
n_samples=(1000)
num_epochs=4
multilen=0

export WANDB_MODE=offline
for n_sample in "${n_samples[@]}"; do
  echo "Evaluating on n_sample = $n_sample"
  for domain in "${domains[@]}"; do
    accelerate launch detectors/detectanyllm/eval.py \
        --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/cross-domain/n${n_sample}/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${domain}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
        --eval_data_path ./data/v1/cross-domain/${domain}_sample.csv \
        --eval_data_format detect \
        --wandb True \
        --train_method ${TRAIN_METHOD} \
        --eval_batch_size 4 \
        --save_dir ./detectors/detectanyllm/results/v1/cross-domain \
        --save_file ${domain}_n${n_sample}_multi${multilen}.json
  done

  for model in "${models[@]}"; do
    accelerate launch detectors/detectanyllm/eval.py \
    --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/cross-model/n${n_sample}/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${model}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
    --eval_data_path ./data/v1/cross-model/${model}_sample.csv \
    --eval_data_format detect \
    --wandb True \
    --train_method ${TRAIN_METHOD} \
    --eval_batch_size 4 \
    --save_dir ./detectors/detectanyllm/results/v1/cross-model \
    --save_file ${model}_n${n_sample}_multi${multilen}.json
  done

  for operation in "${operations[@]}"; do
    accelerate launch detectors/detectanyllm/eval.py \
    --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/cross-operation/n${n_sample}/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${operation}_ep${num_epochs}_n${n_sample}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
    --eval_data_path ./data/v1/cross-operation/${operation}_sample.csv \
    --eval_data_format detect \
    --wandb True \
    --train_method ${TRAIN_METHOD} \
    --eval_batch_size 4 \
    --save_dir ./detectors/detectanyllm/results/v1/cross-operation \
    --save_file ${operation}_n${n_sample}_multi${multilen}.json
  done
done
