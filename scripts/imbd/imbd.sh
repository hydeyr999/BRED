domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
#operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
operations=('translate')
languages=("ch" "de" "fr")
DATANUM=250
EPOCHS=10
threshold="0.01"
LR=1e-4
BETA=0.05
device='cuda:3'

#python detectors/imbd/run_spo.py \
#    --datanum=$DATANUM \
#    --epochs=$EPOCHS \
#    --threshold=$threshold \
#    --lr=$LR \
#    --beta=$BETA \
#    --device=$device \
#    --train_dataset HC3 \
#    --eval_dataset translate \
#    --translate True

#for DATASET in ${domains[@]}; do
#  echo "Processing cross-domain: $DATASET"
#  python detectors/imbd/run_spo.py \
#      --eval_only True \
#      --from_pretrained=./detectors/imbd/ckpt/ai_detection_500/imbd_spo_lr_0.0001_beta_0.05_a_1 \
#      --eval_dataset "$DATASET" \
#      --device $device
#done
#
#for MODEL in ${models[@]}; do
#  echo "Processing cross-model: $MODEL"
#  python detectors/imbd/run_spo.py \
#    --eval_only True \
#    --from_pretrained=./detectors/imbd/ckpt/ai_detection_500/imbd_spo_lr_0.0001_beta_0.05_a_1 \
#    --eval_dataset $MODEL \
#    --device $device
#done

#for OPERATION in ${operations[@]}; do
#  echo "Processing cross-operation: $OPERATION"
#  python detectors/imbd/run_spo.py \
#    --eval_only True \
#    --from_pretrained=./detectors/imbd/ckpt/ai_detection_500/imbd_n$DATANUM\_ep$EPOCHS\_lr_0.0001_beta_0.05_a_1 \
#    --eval_dataset $OPERATION \
#    --device $device
#done

#for domain in ${domains[@]}; do
#  echo "Processing cross-operation: translate on $domain"
#  python detectors/imbd/run_spo.py \
#    --eval_only True \
#    --from_pretrained=./detectors/imbd/ckpt/ai_detection_500/imbd_n$DATANUM\_ep$EPOCHS\_lr_0.0001_beta_0.05_a_1\_False \
#    --eval_dataset translate \
#    --device $device \
#    --subdomain $domain \
#    --epochs $EPOCHS
#done

#translate True
for language in ${languages[@]}; do
#for domain in ${domains[@]}; do
  echo "Processing cross-operation: translate on $domain, language $language"
  python detectors/imbd/run_spo.py \
    --eval_only True \
    --from_pretrained=./detectors/imbd/ckpt/ai_detection_500/HC3_n$DATANUM\_ep$EPOCHS\_lr_0.0001_beta_0.05_a_1\_True \
    --eval_dataset translate \
    --device $device \
    --translate True \
    --epochs $EPOCHS \
    --language $language \
    #    --subdomain $domain \
#done
done
