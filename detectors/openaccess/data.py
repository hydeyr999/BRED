import pandas as pd

def get_df(args):
    if args.multilen>0:
        test_dir = f'./data/v1/multilen/len_{args.multilen}/{args.task}/{args.dataset}_len_{args.multilen}.csv'
        print(test_dir)
        df = pd.read_csv(test_dir)
        df['text'] = df[f'text_{args.multilen}']
    elif args.task == 'llm-co' or args.task == 'op-co':
        test_dir = f'./data/v1/{args.task}/test/{args.dataset}_sample_test_n{args.frac}.json'
        print(test_dir)
        df = pd.read_json(test_dir, lines=True)
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
            path = f'./data/v1/rebuttal/{args.dataset}_sample.csv'
        else:
            path = f'./data/v1/{args.task}/{args.dataset}_sample.csv'
        print(path)
        df = pd.read_csv(path)

    print(df)
    print(df.shape)

    return df