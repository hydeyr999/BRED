domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
TRAIN_METHOD="DDL"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="None"
n_sample=2000
num_epochs=4
multilens=("10" "50" "100" "200" "500")

export WANDB_MODE=offline

for multilen in "${multilens[@]}"; do
for domain in "${domains[@]}"; do
  accelerate launch detectors/detectanyllm/eval.py \
      --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/v1/cross-domain/n2000/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_${domain}_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e4 \
      --eval_data_path ./data/v1/multilen/len_${multilen}/cross-domain/${domain}_len_${multilen}.csv \
      --eval_data_format detect \
      --wandb True \
      --train_method ${TRAIN_METHOD} \
      --eval_batch_size 4 \
      --save_dir ./detectors/detectanyllm/results/v1/cross-domain \
      --save_file ${domain}_n${n_sample}_multi${multilen}.json \
      --multilen ${multilen}
done
done