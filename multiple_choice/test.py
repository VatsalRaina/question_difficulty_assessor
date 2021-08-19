#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import time
import datetime

from transformers import ElectraTokenizer, ElectraForSequenceClassification



MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=24, help='Specify the training batch size')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--predictions_save_path', type=str, help='Load path to which predictions will be saved')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    with open(args.test_data_path + "middle.json") as f:
        middle_data = json.load(f)
    with open(args.test_data_path + "high.json") as f:
        high_data = json.load(f)
    with open(args.test_data_path + "college.json") as f:
        college_data = json.load(f)
    test_data = middle_data + high_data + college_data

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    def asNum(x):
        if x=="A":
            return 0
        if x=="B":
            return 1
        if x=="C":
            return 2
        if x=="D":
            return 3
    
    num_middle = 0
    for item in middle_data:
        num_middle += len(item["questions"])
    num_high = 0
    for item in high_data:
        num_high += len(item["questions"])
    num_college = 0
    for item in college_data:
        num_college += len(item["questions"])

    targets = [0.0]*num_middle + [5.0]*num_high + [10.0]*num_college

    input_ids = []
    token_type_ids = []
    attention_masks = []

    for item in test_data:
        context = item["article"]
        questions = item["questions"]
        for question in questions:
            combo = question + " [SEP] " + context
            input_encodings_dict = tokenizer(combo, truncation=True, max_length=MAXLEN, padding="max_length")
            inp_ids = input_encodings_dict['input_ids']
            inp_att_msk = input_encodings_dict['attention_mask']
            tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]
            input_ids.append(inp_ids)
            token_type_ids.append(tok_type_ids)
            attention_masks.append(inp_att_msk)

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)
    targets = torch.tensor(targets)
    targets = targets.float().to(device)

    print(input_ids.size())
    print(targets.size())

    ds = TensorDataset(input_ids, token_type_ids, attention_masks)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    preds = []

    count = 0
    for inp_id, tok_typ_id, att_msk in dl:
        print(count)
        count+=1
        inp_id, tok_typ_id, att_msk = inp_id.to(device), tok_typ_id.to(device), att_msk.to(device)
        with torch.no_grad():
            outputs = model(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)
        curr_preds = outputs[0]
        preds += curr_preds.detach().cpu().numpy().tolist()
    preds = np.asarray(preds)
    np.save(args.predictions_save_path + "preds_all.npy", preds)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
