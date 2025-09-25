import pandas as pd
import numpy as np
import tqdm


def get_df(args):
    if args.mode == 'train':
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
            raise NotImplementedError
    else:
        if args.multilen>0:
            test_dir = f'./data/v1/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
            df = pd.read_csv(test_dir)
            df['text'] = df[f'text_{args.multilen}']
        elif args.task == 'llm-co' or args.task =='op-co':
            test_dir = f'./data/v1/{args.task}/{args.mode}/{args.dataset}_sample_{args.mode}_n{args.frac}.json'
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
                df1 = pd.DataFrame(
                    {'id': range(len(text1)), 'text': text1, 'generated': generated, 'op1': op1 + op1, 'op2': op2 + op2})
                df2 = pd.DataFrame(
                    {'id': range(len(text2)), 'text': text2, 'generated': generated, 'op1': op1 + op1, 'op2': op2 + op2})
            else:
                model1 = df['model1'].values.tolist()
                model2 = df['model2'].values.tolist()
                df1 = pd.DataFrame(
                    {'id': range(len(text1)), 'text': text1, 'generated': generated, 'model1': model1 + model1,'model2': model2 + model2})
                df2 = pd.DataFrame(
                    {'id': range(len(text2)), 'text': text2, 'generated': generated, 'model1': model1 + model1,'model2': model2 + model2})

            print(df1)
            print(df2)
            return (df1, df2)
        else:
            if args.ori_data_path:
                try:
                    data_df = pd.read_csv(args.ori_data_path).dropna().reset_index(drop=True)
                except:
                    data_df = pd.read_csv(args.ori_data_path,encoding='latin1').dropna().reset_index(drop=True)
                data_df = data_df.rename(columns={
                    data_df.columns[0]: 'human',
                    data_df.columns[1]: 'ai'
                })
                text = data_df['human'].values.tolist() + data_df['ai'].values.tolist()
                generated = [0] * len(data_df) + [1] * len(data_df)
                df = pd.DataFrame(zip(text, generated), columns=['text', 'generated'])
            else:
                test_dir = f'./data/v1/{args.task}'
                path = f'{test_dir}/{args.dataset}_sample.csv'
                print(path)
                df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df


def get_dpic_df(args):
    if args.mode == 'train':
        if args.imbddata:
            if args.translate:
                data_path = f'./detectors/DPIC/dpic_data/otherdata/imbd_35_translate_sample.json'
            else:
                data_path = f'./detectors/DPIC/dpic_data/otherdata/imbd_sample.json'
            df = pd.read_json(data_path,lines=True)
            df = get_sample(df, args.n_sample)
        elif args.task == 'llm-co' or args.task == 'op-co':
            data_path = f'./detectors/DPIC/dpic_data/v1/{args.task}/{args.dataset}_sample_train.json'
            print(data_path)
            df = pd.read_json(data_path,lines=True)
            print(df)
            text = df['text'].values.tolist()
            generated_text = df['dpic_text'].values.tolist()
            generated = df['generated'].values.tolist()
            df = pd.DataFrame(zip(text, generated_text, generated), columns=['text', 'generated_text', 'generated'])
            print(df)
        else:
            datasets_dict = {
                'cross-domain': ['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
                'cross-model': ['llama', 'deepseek', 'gpt4o', 'Qwen'],
                'cross-operation': ['create', 'rewrite', 'summary', 'polish', 'refine', 'expand', 'translate'],
                'rebuttal':['llama', 'llama_8b_instruct', 'deepseek', 'gpt4o', 'gpt4o_large', 'Qwen', 'Qwen_7b','Qwen_8b'],
                'thinking': ['llama', 'deepseek','deepseek_thinking', 'gpt_4o', 'gpt_5','gpt_5_thinking', 'Qwen', 'Qwen_thinking']}
            datasets = datasets_dict[args.task]

            train_dir = f'./detectors/DPIC/dpic_data/v1/{args.task}'
            if args.task == 'rebuttal' or args.task == 'thinking':
                paths = [f'{train_dir}/{x}_sample.json' for x in datasets if args.dataset not in x]
            else:
                paths = [f'{train_dir}/test/{x}_sample.json' for x in datasets if x != args.dataset]
            print(paths)

            datas = []
            for path in tqdm.tqdm(paths):
                original = pd.read_json(path, lines=True)
                original.columns = ['generated_text' if col == 'dpic_text' else col for col in original.columns]
                print(original)
                text = original['text'].values.tolist()
                generated_text = original['generated_text'].values.tolist()
                generated = original['generated'].values.tolist()
                datas.extend(zip(text, generated_text, generated))

            df = pd.DataFrame(datas, columns=['text', 'generated_text', 'generated'])
            df = get_sample(df,args.n_sample)

    elif args.mode == 'test':
        path = f'./detectors/DPIC/dpic_data/v1/{args.task}/{args.dataset}_sample.json'
        print(path)
        if args.task == 'llm-co' or args.task == 'op-co':
            test_dir = f'./detectors/DPIC/dpic_data/v1/{args.task}/{args.dataset}_sample_test_{args.flag}.json'
            print(test_dir)
            df = pd.read_json(test_dir, lines=True)
            df.columns = ['generated_text' if col == 'dpic_text' else col for col in df.columns]
            print(df)
            if args.task == 'op-co':
                print(args.op2)
                df = df[df['op2']==args.op2].reset_index(drop=True)
            elif args.task == 'llm-co':
                print(args.model2)
                df = df[df['model2']==args.model2].reset_index(drop=True)
            else:
                raise ValueError('op2 or model2 is not specified')
        else:
            df = pd.read_json(path, lines=True)
            df.columns = ['generated_text' if col == 'dpic_text' else col for col in df.columns]
        if args.multilen > 0:
            df = run_datasplit(df, args.multilen)
    else:
        raise ValueError(f'Mode is not supported')

    print(df.head())
    print(df.shape)

    return df


def word_split(text):
    try:
        text_list = text.split()
    except Exception as e:
        text_list = None
    return text_list


def safe_join(x):
    if x is None:
        return ''
    else:
        return ' '.join(x)


def data_split(text, max_len):
    words = word_split(text)
    words_cutlen = words[:max_len] if words is not None else None
    text_cutlen = safe_join(words_cutlen)
    return text_cutlen


def run_datasplit(df, max_len):
    df['text'] = df['text'].apply(lambda x: data_split(x, max_len))
    df['generated_text'] = df['generated_text'].apply(lambda x: data_split(x, max_len))
    print(df.head())
    print(df.shape)
    return df


def get_sample(df,n_sample):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    assert len(df_human) == len(df_ai)
    try:
        data_df = pd.DataFrame(
            {'original': df_human['text'], 'original_gen': df_human['generated_text'], 'rewritten': df_ai['text'],
             'rewritten_gen': df_ai['generated_text']}).dropna().reset_index(drop=True)
    except:
        data_df = pd.DataFrame(
            {'original': df_human['text'], 'rewritten': df_ai['text']}).dropna().reset_index(drop=True)
    data_df = data_df.sample(n=n_sample, random_state=12)

    sample_human = data_df['original'].values.tolist()
    sample_ai = data_df['rewritten'].values.tolist()
    sample_text = sample_human + sample_ai
    generated = [0] * len(sample_human) + [1] * len(sample_ai)
    text_id = list(range(len(sample_text)))
    try:
        sample_human_gen = data_df['original_gen'].values.tolist()
        sample_ai_gen = data_df['rewritten_gen'].values.tolist()
        sample_text_gen = sample_human_gen + sample_ai_gen
        sample_df = pd.DataFrame({'id': text_id, 'text': sample_text, 'generated_text': sample_text_gen, 'generated': generated})
    except:
        sample_df = pd.DataFrame({'id': text_id, 'text': sample_text, 'generated': generated})
    return sample_df