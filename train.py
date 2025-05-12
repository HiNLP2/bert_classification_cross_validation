#! -*- coding:utf-8 -*-
# 情感分类任务, 加载bert权重
# valid_acc: 94.72, test_acc: 94.11

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model
from bert4torch.callbacks import Callback, Logger
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything, YamlConfig
from bert4torch.trainer import SequenceClassificationTrainer
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--fold', default='0') #, choices=['0', '1', '2', '3', '4'])  kfold的序号
args = parser.parse_args()
config = YamlConfig('./config.yaml')
fold = args.fold
maxlen = 256
batch_size = 16
model_dir = config['model_dir']
data_dir = os.path.join(config['root_dir'], 'data')
ckpt_dir = os.path.join(config['root_dir'], 'ckpt')
config_path = f'{model_dir}/bert4torch_config.json'
checkpoint_path = f'{model_dir}/pytorch_model.bin'
dict_path = f'{model_dir}/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
choice = 'train'  # train表示训练，infer表示推理

# 固定seed
seed_everything(42)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据，并尽量划分为不超过maxlen的句子
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                token_ids, segment_ids = tokenizer.encode(l['content'], maxlen=maxlen)
                D.append((token_ids, segment_ids, l['class_id']))
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for token_ids, segment_ids, label in batch:
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append(label)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels

# 定义模型
bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True, gradient_checkpoint=True)
model = SequenceClassificationTrainer(bert, num_labels=45).to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
    metrics=['accuracy']
)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = self.evaluate(valid_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(os.path.join(ckpt_dir, f'best_model_{fold}.pt'))
        logs['val_acc'] = val_acc
        logs['best_val_acc'] = self.best_val_acc
        print(f'val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')

    # 定义评价函数
    def evaluate(self, data):
        total, right = 0., 0.
        for x_true, y_true in tqdm(data):
            y_pred = model.predict(x_true).argmax(axis=1)
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        return right / total

def inference(model_predict, texts, batch_size=10):
    '''样本推理
    '''
    result_pred, result_pred_logit = [], []
    for start_index in tqdm(range(0, int(np.ceil(len(texts)/batch_size)*batch_size), batch_size)):
        batch_text = texts[start_index:start_index+batch_size]
        token_ids, segment_ids = tokenizer.encode(batch_text, maxlen=maxlen)
        token_ids = torch.tensor(sequence_padding(token_ids), dtype=torch.long, device=device)
        segment_ids = torch.tensor(sequence_padding(segment_ids), dtype=torch.long, device=device)
        logit = model_predict.predict([token_ids, segment_ids])
        y_pred_logit, y_pred = torch.max(torch.softmax(logit, dim=-1), dim=-1)
        y_pred_logit:torch.Tensor
        y_pred:torch.Tensor
        result_pred.extend(y_pred.cpu().numpy().tolist())
        result_pred_logit.extend(y_pred_logit.cpu().numpy().tolist())
    return result_pred, result_pred_logit


if __name__ == '__main__':
    # 加载数据集
    train_dataloader = DataLoader(MyDataset(f'{data_dir}/fold_{fold}_train.jsonl'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
    valid_dataloader = DataLoader(MyDataset(f'{data_dir}/fold_{fold}_test.jsonl'), batch_size=batch_size, collate_fn=collate_fn) 

    if choice == 'train':
        evaluator = Evaluator()
        logger = Logger(f'{ckpt_dir}/best_model_{fold}.log')
        model.fit(train_dataloader, epochs=10, steps_per_epoch=None, callbacks=[evaluator, logger])
    else:
        model.load_weights('best_model.pt')
        inference(model, ['我今天特别开心', '我今天特别生气'])
