import argparse
from data import *
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_eval(probs_df, df ,model_name):
    roc_auc = roc_auc_score(df.generated.values, probs_df.generated.values)
    print(f'{model_name}_roc_auc:', roc_auc)

    precision, recall, _ = precision_recall_curve(df.generated.values, probs_df.generated.values)
    pr_auc = auc(recall, precision)
    print(f'{model_name}_pr_auc:', pr_auc)

    results_df = pd.DataFrame({'auc': [roc_auc], 'pr': [pr_auc]})
    return results_df


def get_llm_result(args,test_cfg,save_dir,model_id,df,model_name):
    import run_llm_inference

    run_llm_inference.main(test_cfg,save_dir, model_id,df,device=args.device)
    llm_probs_df_m0 = pd.read_parquet(f'{save_dir}/{model_id}.parquet')

    llm_probs_df_m0 = llm_probs_df_m0.sort_values(by='id')
    print(llm_probs_df_m0)

    results_df = get_eval(llm_probs_df_m0,df,model_name)
    results_df.to_csv(f'{save_dir}/{model_id}_results.csv',index=False)

    return llm_probs_df_m0

def train_llm(args,train_df):
    import train_mistral
    mistral_dir = './detectors/mistral'
    train_cfg = OmegaConf.load(f'{mistral_dir}/conf/train.yaml')
    train_cfg.train_params.num_train_epochs = args.epochs
    if args.imbddata:
        train_cfg.outputs.model_dir = f'{mistral_dir}/weights/v1/otherdata/mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}'
    elif args.task == 'op-co' or args.task == 'llm-co':
        train_cfg.outputs.model_dir = f'{mistral_dir}/weights/v1/{args.task}/mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.frac}'
    else:
        train_cfg.outputs.model_dir = f'{mistral_dir}/weights/v1/{args.task}/mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}'
    print(args.device)
    print(train_cfg)
    train_mistral.run_training(train_cfg,train_df,args.device,args.threshold)

def get_llm_test(args,df,flag=None):
    mistral_dir = './detectors/mistral'
    save_dir = f'{mistral_dir}/results/v1/{args.task}'
    test_cfg = OmegaConf.load(f'{mistral_dir}/conf/test.yaml')
    if args.imbddata:
        test_cfg.model.lora_path = f'{mistral_dir}/weights/v1/otherdata/mistral_HC3_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}/best'
        save_name = f'mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_multi{args.multilen}_n{args.n_sample}_{args.translate}'
    elif args.task == 'op-co':
        test_cfg.model.lora_path = f'{mistral_dir}/weights/v1/{args.task}/mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.frac}/best'
        save_name = f'mistral_{args.dataset}_{args.op2}_n{args.frac}_{flag}_ep{args.epochs}_thres{args.threshold}'
    elif args.task == 'llm-co':
        test_cfg.model.lora_path = f'{mistral_dir}/weights/v1/{args.task}/mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.frac}/best'
        save_name = f'mistral_{args.dataset}_{args.model2}_n{args.frac}_{flag}_ep{args.epochs}_thres{args.threshold}'
    elif args.task == 'thinking':
        base_dataset = args.dataset.split('_')[0]
        test_cfg.model.lora_path = f'{mistral_dir}/weights/v1/{args.task}/mistral_{base_dataset}_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}/best'
        print(test_cfg.model.lora_path)
        save_name = f'mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_multi{args.multilen}_n{args.n_sample}'
    else:
        test_cfg.model.lora_path = f'{mistral_dir}/weights/v1/{args.task}/mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_n{args.n_sample}/best'
        save_name = f'mistral_{args.dataset}_ep{args.epochs}_thres{args.threshold}_multi{args.multilen}_n{args.n_sample}'
    print(args.device)
    print(test_cfg)
    get_llm_result(args,test_cfg,save_dir,save_name,df,'mistral')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cross-domain')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['train','test'],default='test')
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wo_domain', type=str, default=None)
    parser.add_argument('--n_sample',type=int,default=2000)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--imbddata',type=bool,default=False)
    parser.add_argument('--frac',type=float,default=0.2)
    parser.add_argument('--op2',type=str,default=None)
    parser.add_argument('--model2',type=str,default=None)
    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        if args.imbddata:
            train_dir = './data/otherdata'
            train_df = pd.read_csv(f'{train_dir}/{args.dataset}_sample.csv')
            train_df['id'] = range(len(train_df))
            print(train_df)
        else:
            train_df = get_df(args).dropna().reset_index(drop=True)
            print(train_df)
        train_llm(args,train_df)
    elif args.mode == 'test':
        if args.task == 'op-co' or args.task == 'llm-co':
            test_df1,test_df2 = get_df(args)
            test_df1 = test_df1.dropna().reset_index(drop=True)
            test_df2 = test_df2.dropna().reset_index(drop=True)
            get_llm_test(args,test_df1,flag='1')
            get_llm_test(args,test_df2,flag='2')
        else:
            test_df = get_df(args).dropna().reset_index(drop=True)
            test_df['id'] = test_df.index
            print(test_df)
            get_llm_test(args,test_df)
