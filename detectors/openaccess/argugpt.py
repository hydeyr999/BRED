import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
from data import *
import tqdm
import torch
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)

def run_mpu(args,df,flag=None):
    tokenizer = AutoTokenizer.from_pretrained("./detectors/opensource/argugpt-sent")
    detector = AutoModelForSequenceClassification.from_pretrained("./detectors/opensource/argugpt-sent").to(args.device)

    # df = get_df(args).dropna().reset_index(drop=True)
    texts = df['text'].values.tolist()
    generated = df['generated'].values.tolist()

    preds = []
    labels = []
    for idx in tqdm.tqdm(range(len(texts))):
        text = texts[idx]
        label = generated[idx]
        try:
            tokenized = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.device)
            with torch.no_grad():
                pred = detector(**tokenized).logits.softmax(-1)[0, 1].item()
            preds.append(pred)
            labels.append(label)
        except Exception as e:
            continue

    _,_,auc_score = get_roc_metrics(labels, preds)
    _,_,pr_score = get_pr_metrics(labels, preds)
    print(f'AUC Score: {auc_score}, PR score: {pr_score}')
    import pandas as pd
    results_df = pd.DataFrame({'auc': [auc_score], 'pr': [pr_score]})
    if args.task == 'op-co':
        results_df.to_csv(f'./detectors/opensource/results/v1/{args.task}/argugpt_{args.dataset}_{args.op2}_n{args.frac}_{flag}.csv', index=False)
    elif args.task == 'llm-co':
        results_df.to_csv(f'./detectors/opensource/results/v1/{args.task}/argugpt_{args.dataset}_{args.model2}_n{args.frac}_{flag}.csv', index=False)
    else:
        results_df.to_csv(f'./detectors/opensource/results/v1/{args.task}/argugpt_{args.dataset}_multi{args.multilen}.csv',index=False)


def get_roc_metrics(real_labels, predictions):
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)
def get_pr_metrics(real_labels, predictions):
    precisions, recalls, thresholds = precision_recall_curve(real_labels, predictions)
    pr_auc = average_precision_score(real_labels, predictions)
    return precisions.tolist(), recalls.tolist(), float(pr_auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['cross-domain', 'cross-operation', 'cross-model'],default='cross-domain')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--max_len',type=int,default=512)
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--language', type=str, default='ch')
    parser.add_argument('--frac',type=float,default=0.2)
    parser.add_argument('--op2',type=str,default=None)
    parser.add_argument('--model2',type=str,default=None)
    args = parser.parse_args()

    if args.task == 'op-co' or args.task == 'llm-co':
        test_df1,test_df2 = get_df(args).dropna().reset_index(drop=True)
        run_mpu(args,test_df1,flag='1')
        run_mpu(args,test_df2,flag='2')
    else:
        test_df = get_df(args).dropna().reset_index(drop=True)
        run_mpu(args,test_df)