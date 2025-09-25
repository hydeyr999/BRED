import pandas as pd
import tqdm
import os

def get_df(args):
    if args.mode =='train':
        if args.imbddata:
            data_path = f'./data/otherdata/imbd_sample.csv'
            print(data_path)
            df = pd.read_csv(data_path)
            print(df)
            # df = get_sample(df,args.n_sample)
        elif args.task == 'llm-co' or args.task == 'op-co':
            data_path = f'./data/v1/{args.task}/train/{args.dataset}_sample_train_n{args.frac}.json'
            print(data_path)
            df = pd.read_json(data_path,lines=True)
            print(df)
            human = df['human'].values.tolist()
            LLM_1 = df['LLM_1'].values.tolist()
            text = human + LLM_1
            generated = [0] * len(human) + [1] * len(LLM_1)
            df = pd.DataFrame({'id': range(len(text)), 'text': text, 'generated': generated})
        else:
            if args.re:
                datasets = ['llama', 'llama_8b_instruct', 'deepseek', 'gpt4o', 'gpt4o_large', 'Qwen', 'Qwen_7b','Qwen_8b']
                train_dir = f'./data/v1/rebuttal'
                paths = [f'{train_dir}/{x}_sample.csv' for x in datasets if args.dataset not in x]
                print(paths)
            elif args.task == 'thinking':
                datasets = ['llama', 'deepseek','deepseek_thinking', 'gpt_4o', 'gpt_5','gpt_5_thinking', 'Qwen', 'Qwen_thinking']
                paths = [f'./data/v1/{args.task}/{x}_sample.csv' for x in datasets if args.dataset not in x]
                print(paths)
            else:
                datasets_dict = {
                    'cross-domain': ['xsum', 'pubmedqa', 'squad', 'writingprompts', 'openreview', 'blog', 'tweets'],
                    'cross-model': ['llama', 'deepseek', 'gpt4o', 'Qwen'],
                    'cross-operation': ['create', 'rewrite', 'summary', 'polish', 'refine', 'expand', 'translate'], }
                datasets = datasets_dict[args.task]
                paths = [f'./data/v1/{args.task}/{x}_sample.csv' for x in datasets if x != args.dataset]
                print(paths)

            datas = []
            for path in tqdm.tqdm(paths):
                original = pd.read_csv(path)
                text = original['text'].values.tolist()
                generated = original['generated'].values.tolist()
                datas.extend(zip(text, generated))

            df = pd.DataFrame(datas, columns=['text', 'generated'])
            df['id'] = range(len(df))
            print(df)

            df = get_sample(df,args.n_sample)
    else:
        if args.multilen>0:
            test_dir = f'./data/v1/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
            df = pd.read_csv(test_dir)
            df['text'] = df[f'text_{args.multilen}']
        elif args.task == 'llm-co' or args.task == 'op-co':
            test_dir = f'./data/v1/{args.task}/test/{args.dataset}_sample_test_n{args.frac}.json'
            print(test_dir)
            df = pd.read_json(test_dir, lines=True)
            if args.op2 is not None:
                print(args.op2)
                df = df[df['op2']==args.op2].reset_index(drop=True)
            elif args.model2 is not None:
                print(args.model2)
                df = df[df['model2']==args.model2].reset_index(drop=True)
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
            if args.re:
                test_dir = f'./data/v1/rebuttal'
            else:
                test_dir = f'./data/v1/{args.task}'
            path = f'{test_dir}/{args.dataset}_sample.csv'
            print(path)
            df = pd.read_csv(path)

    print(df)
    print(df.shape)

    return df

def get_sample(df,n_sample=2000):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    assert len(df_human) == len(df_ai)
    data_df = pd.DataFrame({'original': df_human['text'], 'rewritten': df_ai['text']}).dropna().reset_index(drop=True)
    data_df = data_df.sample(n=n_sample, random_state=12)

    sample_human = data_df['original'].values.tolist()
    sample_ai = data_df['rewritten'].values.tolist()
    sample_text = sample_human + sample_ai
    generated = [0] * len(sample_human) + [1] * len(sample_ai)
    id = list(range(len(sample_text)))
    sample_df = pd.DataFrame({'id': id, 'text': sample_text, 'generated': generated})

    return sample_df