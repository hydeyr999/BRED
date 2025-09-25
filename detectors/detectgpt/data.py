import pandas as pd
import tqdm
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


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
        test_dir = f'./data/v1/{args.task}'
        path = f'{test_dir}/{args.dataset}_sample.csv'
        print(path)
        df = pd.read_csv(path)

    print(df.head())
    print(df.shape)

    return df

def get_model(model_path,device):
    device_map = {'' : int(device.split(':')[1])}
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if model_path == 'tinyllama':
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            use_cache=False,
            device_map=device_map
        )

    return model,tokenizer