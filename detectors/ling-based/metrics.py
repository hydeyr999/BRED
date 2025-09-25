import torch
import torch.nn.functional as F
import tqdm
import gc
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import argparse
import ast
from functions import *
import os
from nltk.tokenize import sent_tokenize
import joblib

def get_model(args):
    device_map = {'' : int(args.device.split(':')[1])}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map
    )

    return model,tokenizer
def get_likelihood(logits, labels, offsets):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    text_ll = log_likelihood.mean().item()
    sent_ll = list()
    for start, end in offsets:
        nll = log_likelihood[start: end].sum() / (end - start)
        sent_ll.append(nll.exp().item())

    max_sent_ll = max(sent_ll)
    sent_ll_avg = sum(sent_ll) / len(sent_ll)
    if len(sent_ll) > 1:
        sent_ll_std = torch.std(torch.tensor(sent_ll)).item()
    else:
        sent_ll_std = 0

    mask = torch.tensor([1] * log_likelihood.size(0)).to(args.device)
    step_ll = log_likelihood.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_ll = step_ll.max(dim=-1)[0].item()
    step_ll_avg = step_ll.sum(dim=-1).div(log_likelihood.size(0)).item()
    if step_ll.size(0) > 1:
        step_ll_std = step_ll.std().item()
    else:
        step_ll_std = 0

    ll = [text_ll, max_sent_ll, sent_ll_avg, sent_ll_std, max_step_ll, step_ll_avg, step_ll_std]

    return ll

def get_rank(logits, labels, offsets):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1

    text_rk = -ranks.mean().item()
    sent_rk = list()
    for start, end in offsets:
        nll = -ranks[start: end].sum() / (end - start)
        sent_rk.append(nll.exp().item())

    max_sent_rk = max(sent_rk)
    sent_rk_avg = sum(sent_rk) / len(sent_rk)
    if len(sent_rk) > 1:
        sent_rk_std = torch.std(torch.tensor(sent_rk)).item()
    else:
        sent_rk_std = 0

    mask = torch.tensor([1] * ranks.size(0)).to(args.device)
    step_rk = -ranks.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_rk = step_rk.max(dim=-1)[0].item()
    step_rk_avg = step_rk.sum(dim=-1).div(ranks.size(0)).item()
    if step_rk.size(0) > 1:
        step_rk_std = step_rk.std().item()
    else:
        step_rk_std = 0

    rk = [text_rk, max_sent_rk, sent_rk_avg, sent_rk_std, max_step_rk, step_rk_avg, step_rk_std]

    return rk


def get_logrank(logits, labels,offsets):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1
    ranks = torch.log(ranks)
    # return -ranks.mean().item()
    text_rk = -ranks.mean().item()
    sent_rk = list()
    for start, end in offsets:
        nll = -ranks[start: end].sum() / (end - start)
        sent_rk.append(nll.exp().item())

    max_sent_rk = max(sent_rk)
    sent_rk_avg = sum(sent_rk) / len(sent_rk)
    if len(sent_rk) > 1:
        sent_rk_std = torch.std(torch.tensor(sent_rk)).item()
    else:
        sent_rk_std = 0

    mask = torch.tensor([1] * ranks.size(0)).to(args.device)
    step_rk = -ranks.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_rk = step_rk.max(dim=-1)[0].item()
    step_rk_avg = step_rk.sum(dim=-1).div(ranks.size(0)).item()
    if step_rk.size(0) > 1:
        step_rk_std = step_rk.std().item()
    else:
        step_rk_std = 0

    rk = [text_rk, max_sent_rk, sent_rk_avg, sent_rk_std, max_step_rk, step_rk_avg, step_rk_std]

    return rk

def get_entropy(logits, labels, offsets):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    # return entropy.mean().item()

    text_en = entropy.mean().item()
    sent_en = list()
    for start, end in offsets:
        nll = entropy[start: end].sum() / (end - start)
        sent_en.append(nll.exp().item())

    max_sent_en = max(sent_en)
    sent_en_avg = sum(sent_en) / len(sent_en)
    if len(sent_en) > 1:
        sent_en_std = torch.std(torch.tensor(sent_en)).item()
    else:
        sent_en_std = 0

    mask = torch.tensor([1] * entropy.size(0)).to(args.device)
    step_en = entropy.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_en = step_en.max(dim=-1)[0].item()
    step_en_avg = step_en.sum(dim=-1).div(entropy.size(0)).item()
    if step_en.size(0) > 1:
        step_en_std = step_en.std().item()
    else:
        step_en_std = 0

    en = [text_en, max_sent_en, sent_en_avg, sent_en_std, max_step_en, step_en_avg, step_en_std]

    return en

def get_baselines(args,model,tokenizer,test_df):
    texts = test_df['text'].values
    generated = test_df['generated'].values

    criterion_fns = {'likelihood': get_likelihood,
                     'rank': get_rank,
                     'logrank': get_logrank,
                     'entropy': get_entropy}
    metrics = {
        'likelihood': [],
        'rank': [],
        'logrank': [],
        'entropy': []
    }
    for name in criterion_fns:
        criterion_fn = criterion_fns[name]
        for text in tqdm.tqdm(texts):
            input_max_length = tokenizer.model_max_length - 2
            token_ids, offsets = list(), list()
            sentences = sent_tokenize(text)

            for s in sentences:
                tokens = tokenizer.tokenize(s)
                ids = tokenizer.convert_tokens_to_ids(tokens)
                difference = len(token_ids) + len(ids) - input_max_length
                if difference > 0:
                    ids = ids[:-difference]
                offsets.append((len(token_ids), len(token_ids) + len(ids)))
                token_ids.extend(ids)
                if difference >= 0:
                    break

            tokenized = tokenizer(text, return_tensors="pt", padding=True,return_token_type_ids=False).to(model.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits = model(**tokenized).logits[:, :-1]
                text_crit = criterion_fn(logits, labels, offsets)
            metrics[name].append(text_crit)

    return metrics

def run(df,flag=None):
    if args.run_metrics:
        model, tokenizer = get_model(args)
        metrics = get_baselines(args, model, tokenizer, df)
        for name in metrics:
            metric = metrics[name]
            df[name] = metric
        print(df)
        if args.save_metrics:
            if args.task == 'op-co' or args.task == 'llm-co':
                if args.mode == 'train':
                    df.to_json(f'./detectors/ling-based/metrics/v1/{args.task}/{args.dataset}_metrics_{args.mode}_n{args.frac}.json',orient='records',lines=True)
                else:
                    df.to_json(f'./detectors/ling-based/metrics/v1/{args.task}/{args.dataset}_metrics_{args.mode}_n{args.frac}_{flag}.json',orient='records',lines=True)
            else:
                df.to_json(f'./detectors/ling-based/metrics/v1/{args.task}/{args.dataset}_metrics_multilen{args.multilen}.json',orient='records',lines=True)
        return

    metrics = {
        'likelihood': df['likelihood'].apply(lambda x: [0 if (i is None or np.isnan(i) or np.isinf(i)) else i for i in x]),
        'rank': df['rank'].apply(lambda x: [0 if (i is None or np.isnan(i) or np.isinf(i)) else i for i in x]),
        'logrank': df['logrank'].apply(lambda x: [0 if (i is None or np.isnan(i) or np.isinf(i)) else i for i in x]),
        'entropy': df['entropy'].apply(lambda x: [0 if (i is None or np.isnan(i) or np.isinf(i)) else i for i in x]),
    }
    for name in metrics:
        metric = metrics[name]
        print(metric)
    generated = np.array(df['generated'].values.tolist())

    if args.mode == 'train':
        features = np.array(metrics[args.func].values.tolist())
        print(features.shape)
        print(generated.shape)
        classifier = get_classifier(features,generated)
        if args.task == 'op-co' or args.task == 'llm-co':
            joblib.dump(classifier,
                        f'./detectors/ling-based/classifier/v1/{args.task}/{args.func}_{args.dataset}_n{args.frac}.joblib')
        else:
            joblib.dump(classifier,
                    f'./detectors/ling-based/classifier/v1/{args.task}/{args.func}_{args.dataset}_n{args.n_sample}_multilen{args.multilen}.joblib')
    else:
        classifier = joblib.load(args.classifier)

        features = np.array(metrics[args.func].values.tolist())
        predictions = classifier.predict_proba(features)[:, 1]

        fpr, tpr, roc_auc = get_roc_metrics(generated, predictions)
        p, r, pr_auc = get_precision_recall_metrics(generated, predictions)
        print(f'ROC AUC: {roc_auc:.4f}', f'\nPR AUC: {pr_auc:.4f}')

        results = {'predictions': predictions.tolist(),
                   'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                   'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                   'loss': 1 - pr_auc}

        res_save_dir = f'./detectors/ling-based/results/v1/{args.task}'
        if not os.path.exists(res_save_dir):
            os.makedirs(res_save_dir)
        if args.task == 'op-co':
            with open(f'{res_save_dir}/{args.func}_{args.dataset}_{args.op2}_n{args.frac}_{args.flag}.json', 'w') as fout:
                json.dump(results, fout)
                print(f'Results written into {res_save_dir}/{args.func}_{args.dataset}_{args.op2}_n{args.frac}_{args.flag}.json')
        elif args.task == 'llm-co':
            with open(f'{res_save_dir}/{args.func}_{args.dataset}_{args.model2}_n{args.frac}_{args.flag}.json', 'w') as fout:
                json.dump(results, fout)
                print(f'Results written into {res_save_dir}/{args.func}_{args.dataset}_{args.model2}_n{args.frac}_{args.flag}.json')
        else:
            with open(f'{res_save_dir}/{args.func}_{args.dataset}_n{args.n_sample}_multi{args.multilen}.json', 'w') as fout:
                json.dump(results, fout)
                print(f'Results written into {res_save_dir}/{args.func}_{args.dataset}_n{args.n_sample}_multi{args.multilen}.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='../detectbt/models/Mistral-7B-v0.1')
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--multilen', type=int, default=0)
    parser.add_argument('--run_metrics',type=bool,default=False)
    parser.add_argument('--save_metrics',type=bool,default=False)
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--func',type=str,default='likelihood')
    parser.add_argument('--n_sample',type=int,default=2000)
    parser.add_argument('--classifier',type=str,default='')
    parser.add_argument('--frac',type=float,default=0.2)
    parser.add_argument('--op2',type=str,default=None)
    parser.add_argument('--model2',type=str,default=None)
    parser.add_argument('--flag',type=str,default='')
    parser.add_argument('--re',type=bool,default=False)
    args = parser.parse_args()
    print(args)

    if args.task == 'op-co' or args.task == 'llm-co':
        if args.run_metrics and args.mode != 'train':
            df1,df2 = get_metrics_df(args)
            df1 = df1.dropna().reset_index(drop=True)
            df2 = df2.dropna().reset_index(drop=True)
            df1 = run(df1,flag='op1')
            df2 = run(df2,flag='op2')
        else:
            df = get_metrics_df(args).dropna().reset_index(drop=True)
            run(df)
    else:
        df = get_metrics_df(args).dropna().reset_index(drop=True)
        run(df)





