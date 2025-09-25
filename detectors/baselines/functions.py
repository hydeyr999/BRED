import pandas as pd
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import random_split

def get_df(args):
    if args.multilen > 0:
        test_dir = f'./data/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
        df = pd.read_csv(test_dir)
        df['text'] = df[f'text_{args.multilen}']
    elif args.task == 'llm-co' or args.task == 'op-co':
        test_dir = f'./data/v1/{args.task}/test/{args.dataset}_sample_test_n{args.frac}.json'
        print(test_dir)
        df = pd.read_json(test_dir, lines=True)
        if args.op2 is not None:
            print(args.op2)
            df = df[df['op2'] == args.op2].reset_index(drop=True)
        elif args.model2 is not None:
            print(args.model2)
            df = df[df['model2'] == args.model2].reset_index(drop=True)
        else:
            raise ValueError('op2 or model2 is not specified')

        human = df['human'].values.tolist()
        LLM_1 = df['LLM_1'].values.tolist()
        LLM_2 = df['LLM_2'].values.tolist()
        text1 = human + LLM_1
        text2 = human + LLM_2
        generated = [0] * len(human) + [1] * len(LLM_2)
        df1 = pd.DataFrame({'id': range(len(text1)), 'text': text1, 'generated': generated})
        df2 = pd.DataFrame({'id': range(len(text2)), 'text': text2, 'generated': generated})

        print(df1)
        print(df2)
        return df1, df2
    else:
        path = f'./data/v1/{args.task}/{args.dataset}_sample.csv'
        df = pd.read_csv(path)
    print(df.head())
    print(df.shape)
    return df


def get_roc_metrics(real_labels, predictions):
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_labels, predictions):
    precision, recall, _ = precision_recall_curve(real_labels, predictions)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)
