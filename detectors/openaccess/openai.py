from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification
import argparse
from data import *
import tqdm
import torch
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)
import logging

def get_openai_pred(detector,tokenizer,text):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    tokenized = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.device)
    with torch.no_grad():
        pred = detector(**tokenized).logits.softmax(-1)[0, 0].item()
    return pred

def get_detect(args,df,flag=None):
    tokenizer = AutoTokenizer.from_pretrained("./detectors/opensource/openai_roberta")
    detector = RobertaForSequenceClassification.from_pretrained("./detectors/opensource/openai_roberta").to(args.device)

    # df = get_df(args).dropna().reset_index(drop=True)
    texts = df['text'].values.tolist()

    preds=[]
    for text in tqdm.tqdm(texts):
        pred = get_openai_pred(detector,tokenizer,text)
        preds.append(pred)

    _, _, auc = get_roc_metrics(df['generated'].values.tolist(), preds)
    _, _, pr_auc = get_precision_recall_metrics(df['generated'].values.tolist(), preds)
    print("AUC: ", auc, "PR AUC: ", pr_auc)
    import pandas as pd
    results_df = pd.DataFrame({'auc': [auc], 'pr_auc': [pr_auc]})
    if args.task == 'op-co':
        results_df.to_csv(f'./detectors/opensource/results/v1/{args.task}/openai_{args.dataset}_{args.op2}_n{args.frac}_{flag}.csv', index=False)
    elif args.task == 'llm-co':
        results_df.to_csv(f'./detectors/opensource/results/v1/{args.task}/openai_{args.dataset}_{args.model2}_n{args.frac}_{flag}.csv', index=False)
    else:
        results_df.to_csv(f'./detectors/opensource/results/v1/{args.task}/openai_{args.dataset}_multi{args.multilen}.csv', index=False)

def get_roc_metrics(real_labels, predictions):
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_labels, predictions):
    precision, recall, _ = precision_recall_curve(real_labels, predictions)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=str,default=None)
    parser.add_argument('--dataset',type=str,default=None)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--device',type=str,default='cuda:2')
    # parser.add_argument('--language', type=str, default='ch')
    parser.add_argument('--frac',type=float,default=0.2)
    parser.add_argument('--op2',type=str,default=None)
    parser.add_argument('--model2',type=str,default=None)
    args = parser.parse_args()
    print(args)
    if args.task == 'op-co' or args.task == 'llm-co':
        test_df1, test_df2 = get_df(args).dropna().reset_index(drop=True)
        get_detect(args, test_df1, flag='1')
        get_detect(args, test_df2, flag='2')
    else:
        test_df = get_df(args).dropna().reset_index(drop=True)
        get_detect(args,test_df)

