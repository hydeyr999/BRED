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

def get_df(args,name_small=None,name_large=None):
    if args.mode == 'train':
        datasets_dict = {
            'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
            'cross-model': ['llama', 'deepseek', 'gpt4o','Qwen'],
            'cross-operation': ['create', 'translate','polish', 'expand','refine','summary','rewrite']}
        if args.run_logprobs == False:
            train_dir = f'./detectors/ling-based/logprobs/v1/{args.task}'
            if args.task == 'op-co' or args.task == 'llm-co':
                path = f'{train_dir}/{args.dataset}_train_n{args.frac}.json'
                print(path)
                df = pd.read_json(path, lines=True)
            else:
                if args.task == 'thinking':
                    datasets = ['llama', 'deepseek', 'deepseek_thinking', 'gpt_4o', 'gpt_5', 'gpt_5_thinking', 'Qwen','Qwen_thinking']
                    paths = [f'{train_dir}/test/{x}_{name_small}_{name_large}.json' for x in datasets if args.dataset not in x]
                    print(paths)
                else:
                    datasets = datasets_dict[args.task]
                    paths = [f'{train_dir}/test/{x}_{name_small}_{name_large}.json' for x in datasets if x != args.dataset]
                    print(paths)
                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_json(path, lines=True)
                    text = original['text'].values.tolist()
                    feat_small = original[f'logprob_small']
                    feat_large = original[f'logprob_large']
                    generated = original['generated'].values.tolist()

                    datas.extend(zip(text, feat_small, feat_large, generated))

                df = pd.DataFrame(datas, columns=['text', f'logprob_small', f'logprob_large', 'generated'])
                print(df)

                df = get_sample(df,args.n_sample)
        else:
            if args.task == 'llm-co' or args.task == 'op-co':
                data_path = f'./data/v1/{args.task}/train/{args.dataset}_sample_train_n{args.frac}.json'
                print(data_path)
                df = pd.read_json(data_path, lines=True)
                print(df)
                human = df['human'].values.tolist()
                LLM_1 = df['LLM_1'].values.tolist()
                text = human + LLM_1
                generated = [0] * len(human) + [1] * len(LLM_1)
                df = pd.DataFrame({'id': range(len(text)), 'text': text, 'generated': generated})
            else:
                datasets = datasets_dict[args.task]
                train_dir = f'./data/{args.task}'
                paths = [f'{train_dir}/{x}_sample.csv' for x in datasets if x != args.dataset]
                print(paths)

                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_csv(path)
                    text = original['text'].values.tolist()
                    generated = original['generated'].values.tolist()
                    datas.extend(zip(text, generated))

                df = pd.DataFrame(datas, columns=['text', 'generated'])
                df = get_sample(df,args.n_sample)
    else:
        if args.run_logprobs == False:
            test_dir = f'./detectors/ling-based/logprobs/v1/{args.task}'
            if args.multilen > 0:
                path = f'{test_dir}/test/{args.dataset}_{name_small}_{name_large}_multi{args.multilen}.json'
                print(path)
                df = pd.read_json(path, lines=True)
            elif args.task == 'op-co' or args.task == 'llm-co':
                path = f'{test_dir}/{args.dataset}_test_n{args.frac}_{args.flag}.json'
                print(path)
                df = pd.read_json(path, lines=True)
                if args.op2 is not None:
                    print(args.op2)
                    df = df[df['op2'] == args.op2].reset_index(drop=True)
                elif args.model2 is not None:
                    print(args.model2)
                    df = df[df['model2'] == args.model2].reset_index(drop=True)
                else:
                    raise ValueError('op2 or model2 is not specified')
            else:
                path = f'{test_dir}/test/{args.dataset}_{name_small}_{name_large}.json'
                print(path)
                df = pd.read_json(path, lines=True)
        else:
            if args.multilen > 0:
                test_dir = f'./data/v1/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
                df = pd.read_csv(test_dir)
                df['text'] = df[f'text_{args.multilen}']
            elif args.task == 'llm-co' or args.task == 'op-co':
                test_dir = f'./data/v1/{args.task}/test/{args.dataset}_sample_test_n{args.frac}.json'
                print(test_dir)
                df = pd.read_json(test_dir, lines=True)
                human = df['human'].values.tolist()
                LLM1 = df['LLM_1'].values.tolist()
                LLM2 = df['LLM_2'].values.tolist()
                text1 = human + LLM1
                text2 = human + LLM2
                generated = [0] * len(human) + [1] * len(LLM1)
                if args.task == 'op-co':
                    op1 = df['op1'].values.tolist()
                    op2 = df['op2'].values.tolist()
                    df1 = pd.DataFrame({'id': range(len(text1)), 'text': text1, 'generated': generated, 'op1': op1 + op1,'op2': op2 + op2})
                    df2 = pd.DataFrame({'id': range(len(text2)), 'text': text2, 'generated': generated, 'op1': op1 + op1,'op2': op2 + op2})
                else:
                    model1 = df['model1'].values.tolist()
                    model2 = df['model2'].values.tolist()
                    df1 = pd.DataFrame({'id': range(len(text1)), 'text': text1, 'generated': generated, 'model1': model1 + model1,'model2': model2 + model2})
                    df2 = pd.DataFrame({'id': range(len(text2)), 'text': text2, 'generated': generated, 'model1': model1 + model1,'model2': model2 + model2})

                print(df1)
                print(df2)
                return(df1,df2)
            else:
                path = f'./data/v1/{args.task}/{args.dataset}_sample.csv'
                print(path)
                df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df

def get_sample(df,n_sample=2000):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    if len(df_human) == len(df_ai):
        try:
            data_df = pd.DataFrame({'text_human': df_human['text'],
                                    'text_ai': df_ai['text'],
                                    'logprob_small_human': df_human['logprob_small'],
                                    'logprob_small_ai': df_ai['logprob_small'],
                                    'logprob_large_human': df_human['logprob_large'],
                                    'logprob_large_ai': df_ai['logprob_large'],
                                    }).dropna().reset_index(drop=True)
        except:
            data_df = pd.DataFrame({'text_human': df_human['text'],
                                    'text_ai': df_ai['text'],
                                    }).dropna().reset_index(drop=True)
    else:
        df = df.sample(n=n_sample*2,random_state=12)
        return df

    data_df = data_df.sample(n=n_sample, random_state=12)
    print(data_df)

    text_human = data_df['text_human'].values.tolist()
    text_ai = data_df['text_ai'].values.tolist()
    text = text_human + text_ai
    generated = [0] * len(text_human) + [1] * len(text_ai)
    id = list(range(len(text)))
    try:
        logprob_small_human = data_df['logprob_small_human'].values.tolist()
        logprob_small_ai = data_df['logprob_small_ai'].values.tolist()
        logprob_large_human = data_df['logprob_large_human'].values.tolist()
        logprob_large_ai = data_df['logprob_large_ai'].values.tolist()
        logprob_small = logprob_small_human + logprob_small_ai
        logprob_large = logprob_large_human + logprob_large_ai
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'logprob_small': logprob_small,
                                  'logprob_large': logprob_large,
                                  'generated': generated})
    except:
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'generated': generated})
    print(sample_df)

    return sample_df

def get_roc_metrics(real_labels, predictions):
    fpr, tpr, _ = roc_curve(real_labels, predictions)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_labels, predictions):
    precision, recall, _ = precision_recall_curve(real_labels, predictions)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_classifier(train_feats, train_labels):
    xgb = XGBClassifier(n_estimators=256, n_jobs=-1)
    lgb = LGBMClassifier(n_estimators=256, n_jobs=-1)
    cat = CatBoostClassifier(n_estimators=256, verbose=0)
    rfr = RandomForestClassifier(n_estimators=256, n_jobs=-1)
    model = VotingClassifier(
        n_jobs=-1,
        voting='soft',
        weights=[4, 5, 4, 4],
        estimators=[('xgb', xgb), ('lgb', lgb), ('cat', cat), ('rfr', rfr)]
    )
    model.fit(train_feats, train_labels)
    return model

def get_gltr_df(args,name_small=None, name_large=None):
    if args.mode == 'train':
        datasets_dict = {
            'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
            'cross-model': ['llama', 'deepseek', 'gpt4o',  'Qwen'],
            'cross-operation': ['create', 'translate', 'polish', 'expand', 'refine', 'summary', 'rewrite']}
        if args.run_feats == False:
            train_dir = f'./detectors/ling-based/lingfeatures/v1/{args.task}'
            if args.task == 'op-co' or args.task == 'llm-co':
                path = f'{train_dir}/{args.dataset}_gltrppl_train_n{args.frac}.json'
                print(path)
                df = pd.read_json(path, lines=True)
            else:
                if args.task == 'thinking':
                    datasets = ['llama', 'deepseek', 'deepseek_thinking', 'gpt_4o', 'gpt_5', 'gpt_5_thinking', 'Qwen','Qwen_thinking']
                    paths = [f'{train_dir}/test/{x}_gltrppl_{name_small}_{name_large}.json' for x in datasets if args.dataset not in x]
                    print(paths)
                else:
                    datasets = datasets_dict[args.task]
                    paths = [f'{train_dir}/test/{x}_gltrppl_{name_small}_{name_large}.json' for x in datasets if x != args.dataset]
                    print(paths)
                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_json(path, lines=True)

                    text = original['text'].values.tolist()
                    gltr_feats = original['gltr_feats']
                    ppl_feats = original['ppl_feats']
                    generated = original['generated'].values.tolist()

                    datas.extend(zip(text, gltr_feats, ppl_feats, generated))

                df = pd.DataFrame(datas, columns=['text', 'gltr_feats', 'ppl_feats', 'generated'])
                print(df)

                df = get_ling_sample(df,args.n_sample)
        else:
            if args.task == 'llm-co' or args.task == 'op-co':
                data_path = f'./data/v1/{args.task}/train/{args.dataset}_sample_train_n{args.frac}.json'
                print(data_path)
                df = pd.read_json(data_path, lines=True)
                print(df)
                human = df['human'].values.tolist()
                LLM_1 = df['LLM_1'].values.tolist()
                text = human + LLM_1
                generated = [0] * len(human) + [1] * len(LLM_1)
                df = pd.DataFrame({'id': range(len(text)), 'text': text, 'generated': generated})
            else:
                train_dir = f'./data/{args.task}'
                datasets = datasets_dict[args.task]
                paths = [f'{train_dir}/{x}_sample.csv' for x in datasets if x != args.dataset]
                print(paths)

                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_csv(path)
                    text = original['text'].values.tolist()
                    generated = original['generated'].values.tolist()
                    datas.extend(zip(text, generated))

                df = pd.DataFrame(datas, columns=['text', 'generated'])
                df = get_ling_sample(df,args.n_sample)
    else:
        if args.run_feats == False:
            test_dir = f'./detectors/ling-based/lingfeatures/v1/{args.task}'
            if args.multilen > 0:
                path = f'{test_dir}/test/{args.dataset}_gltrppl_{name_small}_{name_large}_multi{args.multilen}.json'
                print(path)
                df = pd.read_json(path, lines=True)
            elif args.task == 'op-co' or args.task == 'llm-co':
                path = f'{test_dir}/{args.dataset}_gltrppl_test_n{args.frac}_{args.flag}.json'
                print(path)
                df = pd.read_json(path, lines=True)
                if args.op2 is not None:
                    print(args.op2)
                    df = df[df['op2'] == args.op2].reset_index(drop=True)
                elif args.model2 is not None:
                    print(args.model2)
                    df = df[df['model2'] == args.model2].reset_index(drop=True)
                else:
                    raise ValueError('op2 or model2 is not specified')
            else:
                path = f'{test_dir}/test/{args.dataset}_gltrppl_{name_small}_{name_large}.json'
                print(path)
                df = pd.read_json(path, lines=True)
        else:
            if args.multilen > 0:
                test_dir = f'./data/v1/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
                df = pd.read_csv(test_dir)
                df['text'] = df[f'text_{args.multilen}']
            elif args.task == 'llm-co' or args.task == 'op-co':
                test_dir = f'./data/v1/{args.task}/test/{args.dataset}_sample_test_n{args.frac}.json'
                print(test_dir)
                df = pd.read_json(test_dir, lines=True)
                human = df['human'].values.tolist()
                LLM1 = df['LLM_1'].values.tolist()
                LLM2 = df['LLM_2'].values.tolist()
                text1 = human + LLM1
                text2 = human + LLM2
                generated = [0] * len(human) + [1] * len(LLM1)
                if args.task == 'op-co':
                    op1 = df['op1'].values.tolist()
                    op2 = df['op2'].values.tolist()
                    df1 = pd.DataFrame({'id': range(len(text1)), 'text': text1, 'generated': generated, 'op1': op1 + op1,'op2': op2 + op2})
                    df2 = pd.DataFrame({'id': range(len(text2)), 'text': text2, 'generated': generated, 'op1': op1 + op1,'op2': op2 + op2})
                else:
                    model1 = df['model1'].values.tolist()
                    model2 = df['model2'].values.tolist()
                    df1 = pd.DataFrame({'id': range(len(text1)), 'text': text1, 'generated': generated, 'model1': model1 + model1,'model2': model2 + model2})
                    df2 = pd.DataFrame({'id': range(len(text2)), 'text': text2, 'generated': generated, 'model1': model1 + model1,'model2': model2 + model2})
                print(df1)
                print(df2)
                return(df1,df2)
            else:
                path = f'./data/v1/{args.task}/{args.dataset}_sample.csv'
                print(path)
                df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df

def get_ling_sample(df,n_sample=2000):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    try:
        data_df = pd.DataFrame({'text_human': df_human['text'],
                                'text_ai': df_ai['text'],
                                'gltr_feats_human': df_human['gltr_feats'],
                                'gltr_feats_ai': df_ai['gltr_feats'],
                                'ppl_feats_human': df_human['ppl_feats'],
                                'ppl_feats_ai': df_ai['ppl_feats'],
                                }).dropna().reset_index(drop=True)
    except:
        data_df = pd.DataFrame({'text_human': df_human['text'],
                                'text_ai': df_ai['text']}).dropna().reset_index(drop=True)

    data_df = data_df.sample(n=n_sample, random_state=12)
    print(data_df)

    text_human = data_df['text_human'].values.tolist()
    text_ai = data_df['text_ai'].values.tolist()
    text = text_human + text_ai
    generated = [0] * len(text_human) + [1] * len(text_ai)
    id = list(range(len(text)))
    try:
        gltr_feats_human = data_df['gltr_feats_human'].values.tolist()
        gltr_feats_ai = data_df['gltr_feats_ai'].values.tolist()
        ppl_feats_human = data_df['ppl_feats_human'].values.tolist()
        ppl_feats_ai = data_df['ppl_feats_ai'].values.tolist()
        gltr_feats = gltr_feats_human + gltr_feats_ai
        ppl_feats = ppl_feats_human + ppl_feats_ai
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'gltr_feats': gltr_feats,
                                  'ppl_feats': ppl_feats,
                                  'generated': generated})
    except:
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'generated': generated})
    print(sample_df)

    return sample_df

def get_metrics_df(args,name_small=None,name_large=None):
    if args.mode == 'train':
        datasets_dict = {
            'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
            'cross-model': ['llama', 'deepseek', 'gpt4o', 'Qwen'],
            'cross-operation': ['create', 'translate', 'polish', 'expand', 'refine', 'summary', 'rewrite']}
        if args.run_metrics == False:
            if args.task == 'op-co' or args.task == 'llm-co':
                path = f'./detectors/ling-based/metrics/v1/{args.task}/{args.dataset}_metrics_train_n{args.frac}.json'
                print(path)
                df = pd.read_json(path, lines=True)
            else:
                if args.re:
                    datasets = ['llama', 'llama_8b_instruct', 'deepseek', 'gpt4o', 'gpt4o_large', 'Qwen', 'Qwen_7b','Qwen_8b']
                    train_dir = f'./detectors/ling-based/metrics/v1/rebuttal'
                    paths = [f'{train_dir}/{x}_metrics_multilen{args.multilen}.json' for x in datasets if args.dataset not in x]
                    print(paths)
                elif args.task == 'thinking':
                    datasets = ['llama', 'deepseek','deepseek_thinking', 'gpt_4o', 'gpt_5','gpt_5_thinking', 'Qwen', 'Qwen_thinking']
                    train_dir = f'./detectors/ling-based/metrics/v1/{args.task}'
                    paths = [f'{train_dir}/{x}_metrics_multilen{args.multilen}.json' for x in datasets if args.dataset not in x]
                    print(paths)
                else:
                    train_dir = f'./detectors/ling-based/metrics/v1/{args.task}'
                    datasets = datasets_dict[args.task]
                    paths = [f'{train_dir}/{x}_metrics_multilen{args.multilen}.json' for x in datasets if x != args.dataset]
                    print(paths)
                dfs = []
                for path in tqdm.tqdm(paths):
                    df = pd.read_json(path, lines=True)
                    dfs.append(df)
                df = pd.concat(dfs)

                df = get_sample(df,args.n_sample)
        else:
            if args.task == 'llm-co' or args.task == 'op-co':
                data_path = f'./data/v1/{args.task}/train/{args.dataset}_sample_train_n{args.frac}.json'
                print(data_path)
                df = pd.read_json(data_path, lines=True)
                print(df)
                human = df['human'].values.tolist()
                LLM_1 = df['LLM_1'].values.tolist()
                text = human + LLM_1
                generated = [0] * len(human) + [1] * len(LLM_1)
                df = pd.DataFrame({'id': range(len(text)), 'text': text, 'generated': generated})
            else:
                datasets = datasets_dict[args.task]
                train_dir = f'./data/v1/{args.task}'
                paths = [f'{train_dir}/{x}_sample.csv' for x in datasets if x != args.dataset]
                print(paths)

                datas = []
                for path in tqdm.tqdm(paths):
                    original = pd.read_csv(path)
                    text = original['text'].values.tolist()
                    generated = original['generated'].values.tolist()
                    datas.extend(zip(text, generated))

                df = pd.DataFrame(datas, columns=['text', 'generated'])
                df = get_sample(df,args.n_sample)
    else:
        if args.run_metrics == False:
            test_dir = f'./detectors/ling-based/metrics/v1/{args.task}'
            if args.task == 'op-co' or args.task == 'llm-co':
                path = f'{test_dir}/{args.dataset}_metrics_test_n{args.frac}_{args.flag}.json'
                print(path)
                df = pd.read_json(path, lines=True)
                if args.op2 is not None:
                    print(args.op2)
                    df = df[df['op2'] == args.op2].reset_index(drop=True)
                elif args.model2 is not None:
                    print(args.model2)
                    df = df[df['model2'] == args.model2].reset_index(drop=True)
                else:
                    raise ValueError('op2 or model2 is not specified')
            else:
                if args.re:
                    test_dir = f'./detectors/ling-based/metrics/v1/rebuttal'
                path = f'{test_dir}/{args.dataset}_metrics_multilen{args.multilen}.json'
                print(path)
                df = pd.read_json(path, lines=True)
        else:
            if args.multilen > 0:
                test_dir = f'./data/v1/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
                df = pd.read_csv(test_dir)
                df['text'] = df[f'text_{args.multilen}']
            elif args.task == 'llm-co' or args.task == 'op-co':
                test_dir = f'./data/v1/{args.task}/test/{args.dataset}_sample_test_n{args.frac}.json'
                print(test_dir)
                df = pd.read_json(test_dir, lines=True)
                human = df['human'].values.tolist()
                LLM1 = df['LLM_1'].values.tolist()
                LLM2 = df['LLM_2'].values.tolist()
                text1 = human + LLM1
                text2 = human + LLM2
                generated = [0] * len(human) + [1] * len(LLM1)
                if args.task == 'op-co':
                    op1 = df['op1'].values.tolist()
                    op2 = df['op2'].values.tolist()
                    df1 = pd.DataFrame({'id': range(len(text1)), 'text': text1, 'generated': generated, 'op1': op1 + op1,'op2': op2 + op2})
                    df2 = pd.DataFrame({'id': range(len(text2)), 'text': text2, 'generated': generated, 'op1': op1 + op1,'op2': op2 + op2})
                elif args.task == 'llm-co':
                    model1 = df['model1'].values.tolist()
                    model2 = df['model2'].values.tolist()
                    df1 = pd.DataFrame({'id': range(len(text1)), 'text': text1, 'generated': generated, 'model1': model1 + model1,'model2': model2 + model2})
                    df2 = pd.DataFrame({'id': range(len(text2)), 'text': text2, 'generated': generated, 'model1': model1 + model1,'model2': model2 + model2})
                print(df1)
                print(df2)
                return(df1,df2)
            else:
                path = f'./data/v1/{args.task}/{args.dataset}_sample.csv'
                print(path)
                df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df

def get_metrics_sample(df,n_sample=2000):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    try:
        data_df = pd.DataFrame({'text_human': df_human['text'],
                                'text_ai': df_ai['text'],
                                'likelihood_human': df_human['likelihood'],
                                'likelihood_ai': df_ai['likelihood'],
                                'rank_human': df_human['rank'],
                                'rank_ai': df_ai['rank'],
                                'logrank_human': df_human['logrank'],
                                'logrank_ai': df_ai['logrank'],
                                'entropy_human': df_human['entropy'],
                                'entropy_ai': df_ai['entropy'],
                                }).dropna().reset_index(drop=True)
    except:
        data_df = pd.DataFrame({'text_human': df_human['text'],
                                'text_ai': df_ai['text']}).dropna().reset_index(drop=True)

    data_df = data_df.sample(n=n_sample, random_state=12)
    print(data_df)

    text_human = data_df['text_human'].values.tolist()
    text_ai = data_df['text_ai'].values.tolist()
    text = text_human + text_ai
    generated = [0] * len(text_human) + [1] * len(text_ai)
    id = list(range(len(text)))
    try:
        likelihood_human = data_df['likelihood_human'].values.tolist()
        likelihood_ai = data_df['likelihood_ai'].values.tolist()
        rank_human = data_df['rank_human'].values.tolist()
        rank_ai = data_df['rank_ai'].values.tolist()
        logrank_human = data_df['logrank_human'].values.tolist()
        logrank_ai = data_df['logrank_ai'].values.tolist()
        entropy_human = data_df['entropy_human'].values.tolist()
        entropy_ai = data_df['entropy_ai'].values.tolist()
        likelihood = likelihood_human + likelihood_ai
        rank = rank_human + rank_ai
        logrank = logrank_human + logrank_ai
        entropy = entropy_human + entropy_ai
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'likelihood': likelihood,
                                  'rank': rank,
                                  'logrank': logrank,
                                  'entropy': entropy,
                                  'generated': generated})
    except:
        sample_df = pd.DataFrame({'id': id,
                                  'text': text,
                                  'generated': generated})
    print(sample_df)

    return sample_df