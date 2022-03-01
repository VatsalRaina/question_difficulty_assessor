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

from scipy.special import softmax

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=24, help='Specify the training batch size')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--models_dir', type=str, help='Specify path to directory containing all trained complexity models')

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

    # Choose device
    device = get_default_device()

    with open(args.test_data_path) as f:
        test_data = json.load(f)

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)


    input_ids = []
    token_type_ids = []
    attention_masks = []

    for item in test_data:
        context = item["context"]
        question = item["question"]
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

    ds = TensorDataset(input_ids, token_type_ids, attention_masks)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    models = []
    seeds = [1, 2, 3]

    for seed in seeds:
        model_path = args.models_dir + "electra_seed" + str(seed) + ".pt"
        model = torch.load(model_path, map_location=device)
        model.eval().to(device)
        models.append(model)

    preds = []

    count = 0
    for inp_id, tok_typ_id, att_msk in dl:
        print(count)
        count+=1
        inp_id, tok_typ_id, att_msk = inp_id.to(device), tok_typ_id.to(device), att_msk.to(device)
        curr_preds = []
        with torch.no_grad():
            for model in models:
                outputs = model(input_ids=inp_id, attention_mask=att_msk, token_type_ids=tok_typ_id)
                curr_preds.append( softmax(outputs[0].detach().cpu().numpy(), axis=-1) )
        curr_preds = np.asarray(curr_preds)
        curr_preds = np.mean(curr_preds, axis=0)
        preds += curr_preds.tolist()
    preds = np.asarray(preds)
    preds_mean = np.mean(preds, axis=0)
    complexity_score = 0.0 * preds_mean[0] + 0.5 * preds_mean[1] + 1.0 * preds_mean[2]
    print("Complexity score:", complexity_score)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
