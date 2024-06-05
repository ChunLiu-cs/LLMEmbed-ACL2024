# -*- coding: utf-8 -*-
import os
import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import trange
from datasets import load_dataset
import argparse

def rep_extract(task, mode, device, sents, labels, max_len, step):
    model_id = "/home/linux/hf_model/llama2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "output_hidden_states": True
    }
    model_config = AutoConfig.from_pretrained(model_id, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        device_map=device,
        torch_dtype=torch.float16)
    model.eval()

    sents_reps = []
    # for idx in trange(0, 20, step):
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)        
        sents_batch = sents[idx: idx_end]

        sents_batch_encoding = tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)
        
        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding)

            reps_batch_5L = []
            for layer in range(-1, -6, -1):
                reps_batch_5L.append(torch.mean(batch_outputs.hidden_states[layer], axis=1))    
            reps_batch_5L = torch.stack(reps_batch_5L, axis=1)

        sents_reps.append(reps_batch_5L.cpu())
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
        dataset = load_dataset("/home/linux/dataset/SST-2/sst2.py", trust_remote_code=True)

        sents = dataset['train']['sentence']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels, 128, 90)
        
        sents = dataset['validation']['sentence']
        labels = dataset['validation']['label']
        rep_extract(task, 'test', device, sents, labels, 128, 90)

    elif task == 'mr':
        path = f'/home/linux/dataset/MR/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 3000, 3)

        path = f'/home/linux/dataset/MR/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1500, 7)

    elif task == 'agnews':
        dataset = load_dataset("/home/linux/dataset/AGNews/ag_news.py", trust_remote_code=True)
        
        sents = dataset['train']['text']
        labels = dataset['train']['label']
        rep_extract(task, 'train', device, sents, labels, 256, 40)

        sents = dataset['test']['text']
        labels = dataset['test']['label']
        rep_extract(task, 'test', device, sents, labels, 256, 40)

    elif task == 'r8':
        path = f'/home/linux/dataset/R8/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 1024, 10)

        path = f'/home/linux/dataset/R8/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1024, 10)

    elif task == 'r52':
        path = f'/home/linux/dataset/R52/train.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'train', device, sents, labels, 1024, 10)

        path = f'/home/linux/dataset/R52/test.json'
        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        sents = dataset['text']
        labels = dataset['label']
        rep_extract(task, 'test', device, sents, labels, 1024, 10)
