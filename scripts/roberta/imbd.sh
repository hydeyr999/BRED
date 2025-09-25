domains=('xsum' 'pubmedqa' 'squad' 'writingprompts' 'openreview' 'blog' 'tweets')
models=('llama' 'deepseek' 'gpt4o' 'Qwen')
#operations=('create' 'rewrite' 'summary' 'polish' 'refine' 'expand' 'translate')
operations=('translate')
epochs=10
threshold="0.01"
n_sample=250
multilen=0
device='cuda:0'

#python detectors/deberta/run_roberta.py --dataset HC3 --mode train --device $device \
#--epochs $epochs --threshold $threshold --n_sample $n_sample --imbddata True --translate True


#for domain in ${domains[@]}; do
#  echo "Testing domain: $domain, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
#  python detectors/deberta/run_roberta.py --task cross-domain --dataset $domain --mode test \
#  --roberta_model ./detectors/deberta/weights/otherdata/roberta_imbd_ep$epochs\_thres$threshold\_n$n_sample \
#  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata True
#done
#
#for model in ${models[@]}; do
#  echo "Testing model: $model, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
#  python detectors/deberta/run_roberta.py --task cross-model --dataset $model --mode test \
#  --roberta_model ./detectors/deberta/weights/otherdata/roberta_imbd_ep$epochs\_thres$threshold\_n$n_sample \
#  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata True
#done

#for operation in ${operations[@]}; do
#  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
#  python detectors/deberta/run_roberta.py --task cross-operation --dataset $operation --mode test \
#  --roberta_model ./detectors/deberta/weights/otherdata/roberta_imbd_ep$epochs\_thres$threshold\_n$n_sample\_False \
#  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata True
#done

for operation in ${operations[@]}; do
  echo "Testing operation: $operation, epochs: $epochs, threshold: $threshold, n_sample: $n_sample"
  python detectors/deberta/run_roberta.py --task cross-operation --dataset $operation --mode test \
  --roberta_model ./detectors/deberta/weights/otherdata/roberta_HC3_ep$epochs\_thres$threshold\_n$n_sample\_True \
  --device $device --epochs $epochs --threshold $threshold --n_sample $n_sample --multilen $multilen --imbddata True --translate True
done