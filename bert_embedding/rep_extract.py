# -*- coding: utf-8 -*-
import os
import torch
import json
from transformers import BertTokenizer, BertModel
from tqdm import trange
from datasets import load_dataset
import argparse

def rep_extract(task, mode, device, sents, labels):
    model_path = '/home/linux/hf_model/bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path).to(device)
    model.eval()

    max_len = 512
    sents_reps = []
    step = 512
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)        
        sents_batch = sents[idx: idx_end]

        sents_batch_encoding = tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)
        
        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding)
            reps_batch = batch_outputs.pooler_output    
        sents_reps.append(reps_batch.cpu())
    sents_reps = torch.cat(sents_reps)
    
    for idx in range(len(labels)):
        labels[idx] = torch.tensor(labels[idx])
    labels = torch.stack(labels)
    
    print(sents_reps.shape)
    print(labels.shape)
    path = f'{task}/dataset_tensor/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(sents_reps.to('cpu'), path + f'{mode}_sents.pt')
    torch.save(labels, path + f'{mode}_labels.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cuda_no', type=int)
    parser.add_argument('task', type=str)   # sst2, mr, agnews, r8, r52
    args = parser.parse_args()
    device = f'cuda:{args.cuda_no}'
    task = args.task
    
    if task == 'sst2':
        dataset = load_dataset("/home/linux/dataset/SST-2/sst2.py")
        
        sents = dataset['train']['sentence']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels)
        
        sents = dataset['validation']['sentence']
        labels = dataset['validation']['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'mr':
        path = f'/home/linux/dataset/MR/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels)

        path = f'/home/linux/dataset/MR/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'agnews':
        dataset = load_dataset("/home/linux/dataset/AGNews/ag_news.py")
        
        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'r8':
        path = f'/home/linux/dataset/R8/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels)

        path = f'/home/linux/dataset/R8/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels)

    elif task == 'r52':
        path = f'/home/linux/dataset/R52/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels)

        path = f'/home/linux/dataset/R52/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels)
