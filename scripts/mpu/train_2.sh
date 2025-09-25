domains=("xsum" "pubmedqa" "squad" "writingprompts" "openreview" "blog" "tweets")
models=("llama" "deepseek" "gpt4o" "Qwen")
operations=("create" "rewrite" "summary" "polish" "refine" "expand" "translate")
n_sample=2000
epochs=4
threshold="0.0"
device="cuda:2"

for domain in "${domains[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python detectors/mpu/train.py --batch-size 4 --max-sequence-length 512 \
    --task cross-domain --dataset ${domain} --val-data-file v1/cross-domain/${domain}_sample.csv \
    --model-name roberta-large --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun \
    --len_thres 55 --aug_min_length 1 --max-epochs $epochs --weight-decay 0 --mode original_single \
    --aug_mode sentence_deletion-0.25 --clean 1 --learning-rate 5e-05 --seed 0 \
    --n_sample $n_sample --threshold $threshold
done

for model in "${models[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python detectors/mpu/train.py --batch-size 4 --max-sequence-length 512 \
    --task cross-model --dataset ${model} --val-data-file v1/cross-model/${model}_sample.csv \
    --model-name roberta-large --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun \
    --len_thres 55 --aug_min_length 1 --max-epochs $epochs --weight-decay 0 --mode original_single \
    --aug_mode sentence_deletion-0.25 --clean 1 --learning-rate 5e-05 --seed 0 \
    --n_sample $n_sample --threshold $threshold
done

for operation in "${operations[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python detectors/mpu/train.py --batch-size 4 --max-sequence-length 512 \
    --task cross-operation --dataset ${operation} --val-data-file v1/cross-operation/${operation}_sample.csv \
    --model-name roberta-large --local-data data --lamb 0.4 --prior 0.2 --pu_type dual_softmax_dyn_dtrun \
    --len_thres 55 --aug_min_length 1 --max-epochs $epochs --weight-decay 0 --mode original_single \
    --aug_mode sentence_deletion-0.25 --clean 1 --learning-rate 5e-05 --seed 0 \
    --n_sample $n_sample --threshold $threshold
done