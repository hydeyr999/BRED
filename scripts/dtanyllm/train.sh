# main experiment
domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
TRAIN_METHOD="DDL"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="None"
n_sample=1000
num_epochs=4

#accelerate launch detectors/detectanyllm/train.py \
#    --scoring_model_name ${SCORING_MODEL} \
#    --reference_model_name ${REFERENCE_MODEL} \
#    --wandb True \
#    --train_data_path ./detectors/detectanyllm/data/ai_detection_500_polish.raw_data.json \
#    --train_data_format ImBD \
#    --eval_data_path ./detectors/detectanyllm/data/MIRAGE_BENCH/DIG/polish.json \
#    --save_freq 2 \
#    --train_method ${TRAIN_METHOD} \
#    --eval_batch_size 4

export WANDB_MODE=offline
for domain in ${domains[@]}; do
  accelerate launch detectors/detectanyllm/train.py \
      --scoring_model_name ${SCORING_MODEL} \
      --reference_model_name ${REFERENCE_MODEL} \
      --wandb True \
      --eval True \
      --task cross-domain\
      --train_data_path ./data/v1/cross-domain/${domain}_sample.csv \
      --eval_data_path ./data/v1/cross-domain/${domain}_sample.csv \
      --train_data_format detect \
      --eval_data_format detect \
      --save_directory ./detectors/detectanyllm/ckpt/v1/cross-domain/ \
      --wandb_dir ./detectors/detectanyllm/log/v1/cross-domain/ \
      --train_method ${TRAIN_METHOD} \
      --eval_batch_size 4 \
      --n_sample ${n_sample} \
      --num_epochs ${num_epochs}
done

for model in ${models[@]}; do
  accelerate launch detectors/detectanyllm/train.py \
      --scoring_model_name ${SCORING_MODEL} \
      --reference_model_name ${REFERENCE_MODEL} \
      --wandb True \
      --eval True \
      --task cross-model\
      --train_data_path ./data/v1/cross-model/${model}_sample.csv \
      --eval_data_path ./data/v1/cross-model/${model}_sample.csv \
      --train_data_format detect \
      --eval_data_format detect \
      --save_directory ./detectors/detectanyllm/ckpt/v1/cross-model/ \
      --wandb_dir ./detectors/detectanyllm/log/v1/cross-model/ \
      --train_method ${TRAIN_METHOD} \
      --eval_batch_size 4 \
      --n_sample ${n_sample} \
      --num_epochs ${num_epochs}
done

for operation in ${operations[@]}; do
  accelerate launch detectors/detectanyllm/train.py \
  --scoring_model_name ${SCORING_MODEL} \
  --reference_model_name ${REFERENCE_MODEL} \
  --wandb True \
  --eval True \
  --task cross-operation\
  --train_data_path ./data/v1/cross-operation/${operation}_sample.csv \
  --eval_data_path ./data/v1/cross-operation/${operation}_sample.csv \
  --train_data_format detect \
  --eval_data_format detect \
  --save_directory ./detectors/detectanyllm/ckpt/v1/cross-operation/ \
  --wandb_dir ./detectors/detectanyllm/log/v1/cross-operation/ \
  --train_method ${TRAIN_METHOD} \
  --eval_batch_size 4 \
  --n_sample ${n_sample} \
  --num_epochs ${num_epochs}
done