# Evaluate main experiment
# Make sure you have downloaded the checkpoint or ran the train.sh first

TRAIN_METHOD="DDL"
SCORING_MODEL="gpt-neo-2.7B"
REFERENCE_MODEL="None"

export WANDB_MODE=offline
#for SUBSET in "DIG" "SIG"
#do
#    for TASK in "generate" "polish" "rewrite"
#    do
#    accelerate launch eval.py \
#        --pretrained_model_name_or_path ./ckpt/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_ai_detection_500_polish.raw_data_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e5 \
#        --eval_data_path ./data/MIRAGE_BENCH/${SUBSET}/${TASK}.json \
#        --wandb True \
#        --train_method ${TRAIN_METHOD} \
#        --eval_batch_size 4 \
#        --save_dir ./results/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_ai_detection_500_polish.raw_data \
#        --save_file MIRAGE_${SUBSET}_${TASK}.json
#    done
#done

accelerate launch detectors/detectanyllm/eval.py \
    --pretrained_model_name_or_path ./detectors/detectanyllm/ckpt/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_hc3_35_translate_lr0.0001_bs1_rewTgt100.0_oriTgt0.0_r8_e5 \
    --eval_data_path ./data/cross-operation/translate_sample_ch.json \
    --wandb True \
    --train_method ${TRAIN_METHOD} \
    --eval_batch_size 4 \
    --save_dir ./detectors/detectanyllm/results/${TRAIN_METHOD}_score_${SCORING_MODEL}_ref_${REFERENCE_MODEL}_hc3_35_translate \
    --save_file MIRAGE_translate_sample_ch.json \
#    --eval_data_format ImBD