import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Salesforce/codegen25-7b-multi")
parser.add_argument("--chunk_csv_path", type=str, default="a.csv")
parser.add_argument("--output_path", type=str, default="out.pt")

def parse_folder(s):
    kw = 'CsiIpBlock'
    if kw not in s:
        return 'API'
    names = s.split('/')
    return names[names.index(kw)+1]

def get_chunks_by_folder(args):
    df = pd.read_csv(args.chunk_csv_path)
    df['folder'] = df['file'].map(parse_folder)
    print(df.groupby('folder').sum('length').sort_values('length'))
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path, use_auth_token=True, trust_remote_code=True)
    
    val_folders = ['IpIommuGen4']
    train_chunks = defaultdict(list)
    val_chunks = defaultdict(list)
    for folder, code in tqdm(df[['folder', 'code']].values):
        input_ids = tokenizer(code, truncation=False)['input_ids'] + [tokenizer.eos_token_id]
        if folder in val_folders:
            val_chunks[folder].extend(input_ids)
        else:
            train_chunks[folder].extend(input_ids)

    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        'train': train_chunks,
        'val': val_chunks,
        'tokenizer_name': tokenizer.name_or_path,
    }, args.output_path)
    print("Saved at", args.output_path)

args = parser.parse_args()
get_chunks_by_folder(args)