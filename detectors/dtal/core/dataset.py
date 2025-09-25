from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import tqdm

def get_df(args,mode,data_path):
    if mode =='train':
        if args.imbddata:
            df = pd.read_csv(data_path)
            print(df)
            # df = get_sample(df,args.n_sample)
        elif args.task == 'llm-co' or args.task == 'op-co':
            print(data_path)
            df = pd.read_json(data_path,lines=True)
            print(df)
            human = df['human'].values.tolist()
            LLM_1 = df['LLM_1'].values.tolist()
            df = pd.DataFrame({'original': human,'rewritten': LLM_1})
        else:
            datasets_dict = {
                # 'cross-domain': ['xsum','pubmedqa', 'squad',  'writingprompts', 'openreview', 'blog', 'tweets'],
                'cross-domain':['xsum', 'writingprompts', 'squad', 'pubmedqa', 'openreview', 'blog', 'tweets'],
                'cross-model': ['llama', 'deepseek', 'gpt4o','Qwen'],
                'cross-operation': ['create', 'translate', 'polish','expand','refine','summary','rewrite']}
            if args.re:
                datasets = ['llama', 'llama_8b_instruct', 'deepseek', 'gpt4o', 'gpt4o_large', 'Qwen', 'Qwen_7b','Qwen_8b']
                eval_dataset = data_path.split('/')[-1].split('_')[0]
                paths = [f'./data/v1/rebuttal/{x}_sample.csv' for x in datasets if eval_dataset not in x]
                print(paths)
            elif args.task == 'thinking':
                datasets = ['llama', 'deepseek', 'deepseek_thinking', 'gpt_4o', 'gpt_5', 'gpt_5_thinking', 'Qwen', 'Qwen_thinking']
                eval_dataset = data_path.split('/')[-1].split('_')[0]
                paths = [f'./data/v1/thinking/{x}_sample.csv' for x in datasets if eval_dataset not in x]
                print(paths)
            else:
                datasets = datasets_dict[args.task]
                eval_dataset = data_path.split('/')[-1].split('_')[0]
                paths = [f'./data/v1/{args.task}/{x}_sample.csv' for x in datasets if x != eval_dataset]
                print(paths)

            datas = []
            for path in tqdm.tqdm(paths):
                original = pd.read_csv(path)
                text = original['text'].values.tolist()
                generated = original['generated'].values.tolist()
                datas.extend(zip(text, generated))

            df = pd.DataFrame(datas, columns=['text', 'generated'])
            df['id'] = range(len(df))

            df = get_sample(df,mode,args.n_sample)
    else:
        if args.task == 'llm-co' or args.task == 'op-co':
            df = pd.read_json(data_path, lines=True)
            if args.task == 'op-co':
                print(args.op2)
                df = df[df['op2'] == args.op2].reset_index(drop=True)
            elif args.task == 'llm-co':
                print(args.model2)
                df = df[df['model2'] == args.model2].reset_index(drop=True)
            else:
                raise ValueError('op2 or model2 is not specified')

            human = df['human'].values.tolist()
            LLM_1 = df['LLM_1'].values.tolist()
            LLM_2 = df['LLM_2'].values.tolist()
            df1 = pd.DataFrame({'original': human, 'rewritten': LLM_1})
            df2 = pd.DataFrame({'original': human, 'rewritten': LLM_2})
            print(df1)
            print(df2)

            if args.flag == '1':
                return df1
            elif args.flag == '2':
                return df2

        else:
            print(data_path)
            df = pd.read_csv(data_path)
            if args.multilen > 0:
                df['text'] = df[f'text_{args.multilen}']
            df = get_sample(df,mode)

    print(df)
    print(df.shape)

    return df

def get_sample(df,mode,n_sample=2000):
    df_human = df[df['generated'] == 0].reset_index(drop=True)
    df_ai = df[df['generated'] == 1].reset_index(drop=True)
    assert len(df_human) == len(df_ai)
    data_df = pd.DataFrame({'original': df_human['text'], 'rewritten': df_ai['text']}).dropna().reset_index(drop=True)
    if mode == 'train':
        data_df = data_df.sample(n=n_sample, random_state=12).reset_index(drop=True)

    return data_df

class CustomDataset(Dataset):
    def __init__(self,
                 args,
                 data_path,
                 scoring_tokenizer,
                 reference_tokenizer=None,
                 data_format='MIRAGE',
                 mode = 'train'):
        super().__init__()
        # self.data = json.load(open(data_path, 'r'))
        self.scoring_tokenizer = scoring_tokenizer
        self.reference_tokenizer = reference_tokenizer
        self.data_format = data_format
        self.mode = mode
        if self.data_format == 'detect':
            self.data = get_df(args,self.mode,data_path)
        else:
            self.data = json.load(open(data_path, 'r'))

    def __getitem__(self, index):
        if self.data_format == 'MIRAGE':
            original_text = self.data[index]['original']
            rewritten_text = self.data[index]['rewritten']
        else:
            original_text = self.data['original'][index]
            rewritten_text = self.data['rewritten'][index]
        return {
            'original': original_text,
            'rewritten': rewritten_text
        }
    
    def collate_fn(self, batch):
        original_texts = [item['original'] for item in batch]
        rewritten_texts = [item['rewritten'] for item in batch]

        # Tokenize batches with padding and truncation
        original_tokens_for_scoring_model = self.scoring_tokenizer(
            original_texts, return_tensors="pt", padding=True, max_length = 512,truncation=True, return_token_type_ids=False
        )
        rewritten_tokens_for_scoring_model = self.scoring_tokenizer(
            rewritten_texts, return_tensors="pt", padding=True,max_length = 512, truncation=True, return_token_type_ids=False
        )
        original_tokens_for_reference_model = self.reference_tokenizer(
            original_texts, return_tensors="pt", padding=True, max_length = 512,truncation=True, return_token_type_ids=False
        ) if self.reference_tokenizer is not None else {'input_ids': None, 'attention_mask': None}
        rewritten_tokens_for_reference_model = self.reference_tokenizer(
            rewritten_texts, return_tensors="pt", padding=True, max_length = 512,truncation=True, return_token_type_ids=False
        ) if self.reference_tokenizer is not None else {'input_ids': None, 'attention_mask': None}

        return {
            'scoring':{
                'original_input_ids': original_tokens_for_scoring_model["input_ids"],
                'original_attention_mask': original_tokens_for_scoring_model["attention_mask"],
                'rewritten_input_ids': rewritten_tokens_for_scoring_model["input_ids"],
                'rewritten_attention_mask': rewritten_tokens_for_scoring_model["attention_mask"]
            },
            'reference':{
                'original_input_ids': original_tokens_for_reference_model["input_ids"],
                'original_attention_mask': original_tokens_for_reference_model["attention_mask"],
                'rewritten_input_ids': rewritten_tokens_for_reference_model["input_ids"],
                'rewritten_attention_mask': rewritten_tokens_for_reference_model["attention_mask"]
            }
        }
    
    def __len__(self):
        return len(self.data) if self.data_format == 'MIRAGE' else len(self.data['original'])