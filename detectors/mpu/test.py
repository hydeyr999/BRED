import argparse

from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
import json
from data import *
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)


English_model = pipeline('text-classification', model = "./mpu_env2/")

def MPU_method(text):
    try:
        result_dict = English_model(text)
        if result_dict and 'score' in result_dict[0]:
            return result_dict
        else:
            raise ValueError("未能获取预期的结果格式")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        pass

def run_mpu(args):
    df = get_df(args)
    texts = df['text'].values.tolist()

    for text in texts:
        result = MPU_method(text)
        print(result)
        stop


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['cross-domain', 'cross-operation', 'cross-model'],default='cross-domain')
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()

    run_mpu(args)