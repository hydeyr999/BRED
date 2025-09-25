import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../detectbt/models/llama3-8b-v2')
    parser.add_argument('--backbone', type=str, default='../detectbt/models/deberta-v3-large')
    parser.add_argument('--dpic_ckpt', type=str, default=None)
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--task',type=str,default='cross-domain')
    parser.add_argument('--dataset',type=str,default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--multilen',type=int,default=0)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--model_save_dir', type=str, default='./detectors/DPIC/weights/v1')
    parser.add_argument('--data_save_dir', type=str, default='./detectors/DPIC/dpic_data/v1')
    parser.add_argument('--run_generate', type=bool, default=False)
    parser.add_argument('--ori_data_path',type=str, default=None)
    parser.add_argument('--gen_save_path', type=str, default=None)
    parser.add_argument('--n_sample',type=int, default=250)
    parser.add_argument('--imbddata',type=bool, default=False)
    parser.add_argument('--translate',type=bool, default=False)
    parser.add_argument('--threshold',type=float, default=0)
    parser.add_argument('--frac',type=float, default=0.2)
    parser.add_argument('--flag',type=str, default='1')
    parser.add_argument('--op2',type=str, default='')
    parser.add_argument('--model2',type=str, default='')
    args = parser.parse_args()
    print(args)

    if args.mode == 'train':
        from dpic_train import *
        if args.run_generate:
            train_df = generate.run(args)
            if args.gen_save_path:
                train_df.to_json(args.gen_save_path, orient='records', lines=True)
            else:
                train_df.to_json(f'{args.data_save_dir}/{args.task}/{args.dataset}_sample_train.json', orient='records',
                                lines=True)
        run_dpic_train(args)

    elif args.mode == 'test':
        from dpic_test import *
        if args.run_generate:
            if (args.task == 'op-co' or args.task == 'llm-co') and args.mode != 'train':
                test_df1, test_df2 = generate.run(args)
                test_df1.to_json(f'{args.data_save_dir}/{args.task}/{args.dataset}_sample_test_1.json', orient='records',lines=True)
                test_df2.to_json(f'{args.data_save_dir}/{args.task}/{args.dataset}_sample_test_2.json', orient='records',lines=True)
            else:
                test_df = generate.run(args)
                if args.gen_save_path:
                    test_df.to_json(args.gen_save_path, orient='records', lines=True)
                else:
                    test_df.to_json(f'{args.data_save_dir}/{args.task}/{args.dataset}_sample.json', orient='records',
                                    lines=True)
        else:
            test_df = get_dpic_df(args).dropna().reset_index(drop=True)
        run_dpic_test(args,test_df)
