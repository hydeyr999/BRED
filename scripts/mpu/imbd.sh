domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
languages=("ch" "de" "fr")
n_sample=250
epochs=4
threshold="0.0"
device="cuda:1"

CUDA_VISIBLE_DEVICES=0 python detectors/mpu/train.py --batch-size 4 --max-sequence-length 512 \
  --train-data-file otherdata/HC3_35_translate_sample.csv --val-data-file otherdata/HC3_35_translate_sample.csv \
  --model-name roberta-large --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun \
  --len_thres 55 --aug_min_length 1 --max-epochs $epochs --weight-decay 0 --mode original_single \
  --aug_mode sentence_deletion-0.25 --clean 1 --learning-rate 5e-05 --seed 0 \
  --n_sample $n_sample --threshold $threshold

#for operation in ${operations[@]}; do
#  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
#  python detectors/opensource/mpu.py --task cross-operation --dataset $operation --device $device \
#  --mpu_model ./detectors/mpu/results/otherdata/mpu_HC3_ep$epochs\_thres$threshold\_n$n_sample\_False/complete-$epochs
#done

for language in ${languages[@]}; do
for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/opensource/mpu.py --task cross-operation --dataset $operation --device $device \
  --mpu_model ./detectors/mpu/results/otherdata/mpu_HC3_ep$epochs\_thres$threshold\_n$n_sample\_True/complete-$epochs \
  --language $language
done
done