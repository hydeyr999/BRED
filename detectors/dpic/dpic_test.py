import generate
from model import *
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch
import os
import tqdm
from data import *
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
def get_eval(args,probs_df, df ,model_name):
    roc_auc = roc_auc_score(df.generated.values, probs_df.generated.values)
    print(f'{model_name}_roc_auc:', roc_auc)

    precision, recall, _ = precision_recall_curve(df.generated.values, probs_df.generated.values)
    pr_auc = auc(recall, precision)
    print(f'{model_name}_pr_auc:', pr_auc)

    results_df = pd.DataFrame({'auc': [roc_auc], 'pr': [pr_auc]})
    results_df.to_csv(
        f'./detectors/DPIC/results/v1/{args.task}/{model_name}_{args.dataset}_ep{args.epochs}_n{args.n_sample}_multi{args.multilen}_results.csv',
        index=False)


def get_prediction(model,tokenizer, df,device='cuda',max_len=768,batch_size=16):
    model.to(device)
    model.eval()

    test_dataset = AIDataset(df, tokenizer, max_len)
    test_generator = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False)

    pred_prob = np.zeros((len(df),), dtype=np.float32)

    for j, (input_ids_ori, input_ids_gen, attention_mask, _) in tqdm.tqdm(enumerate(test_generator),total=len(test_generator)):
        with torch.no_grad():
            start = j * batch_size
            end = start + batch_size
            if j == len(test_generator) - 1:
                end = len(test_generator.dataset)

            input_ids_ori = input_ids_ori.to(device)
            input_ids_gen = input_ids_gen.to(device)
            attention_mask = attention_mask.to(device)

            with autocast():
                logits = model(input_ids_ori, input_ids_gen, attention_mask)
            pred_prob[start:end] = logits.sigmoid().cpu().data.numpy().squeeze()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return pred_prob

def get_test_result(args,model,tokenizer, df, output_dir):
    bert_prob0 = get_prediction(model,
                                tokenizer,
                                df,
                                max_len=args.max_len,
                                device=args.device)

    # print(bert_prob0)
    gc.collect()
    torch.cuda.empty_cache()

    bert_probs_df = pd.DataFrame(data={'id': df.id.values, 'generated': bert_prob0})
    bert_probs_df.to_csv(output_dir, index=False)
    bert_probs_df = pd.read_csv(output_dir)
    # print('bert_probs_df:', bert_probs_df)

    get_eval(args,bert_probs_df, df,'dpic')


def run_dpic_test(args,test_df):
    model_path = args.backbone
    model_weights = args.dpic_ckpt
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = DPICModel(model_path, config, tokenizer, pretrained=False)
    model.load_state_dict(torch.load(model_weights))

    if args.imbddata:
        output_dir = f'./detectors/DPIC/results/otherdata'
        get_test_result(args, model, tokenizer, test_df,
                        f'{output_dir}/dpic_{args.dataset}_ep{args.epochs}_n{args.n_sample}_multi{args.multilen}_{args.translate}.csv')
    elif args.task == 'op-co':
        output_dir = f'./detectors/DPIC/results/v1/{args.task}'
        get_test_result(args, model, tokenizer, test_df,
                        f'{output_dir}/dpic_{args.dataset}_{args.op2}_{args.flag}.csv')
    elif args.task == 'llm-co':
        output_dir = f'./detectors/DPIC/results/v1/{args.task}'
        get_test_result(args, model, tokenizer, test_df,
                        f'{output_dir}/dpic_{args.dataset}_{args.model2}_{args.flag}.csv')
    else:
        output_dir = f'./detectors/DPIC/results/v1/{args.task}'
        get_test_result(args,model,tokenizer, test_df,
                        f'{output_dir}/dpic_{args.dataset}_ep{args.epochs}_n{args.n_sample}_multi{args.multilen}.csv')