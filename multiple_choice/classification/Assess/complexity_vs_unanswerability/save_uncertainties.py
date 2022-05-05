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

from transformers import ElectraTokenizer, ElectraForMultipleChoice
from keras.preprocessing.sequence import pad_sequences

from Uncertainty import ensemble_uncertainties_classification

from scipy.special import softmax
from scipy.stats import entropy

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=24, help='Specify the evaluation batch size')
parser.add_argument('--gen_questions_path', type=str, help='Load path of test data')
parser.add_argument('--qa_models_dir', type=str, help='Specify path to directory containing all trained complexity models')


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

    with open(args.gen_questions_path) as f:
        test_data = json.load(f)

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    input_ids = []
    token_type_ids = []

    for ex in test_data:
        question, context, options = ex['question'], ex['context'], ex['options']
        four_inp_ids = []
        four_tok_type_ids = []
        for opt in options:
            combo = context + " [SEP] " + question + " " + opt
            inp_ids = tokenizer.encode(combo)
            if len(inp_ids)>512:
                inp_ids = [inp_ids[0]] + inp_ids[-511:]
            tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]
            four_inp_ids.append(inp_ids)
            four_tok_type_ids.append(tok_type_ids)
        four_inp_ids = pad_sequences(four_inp_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        four_tok_type_ids = pad_sequences(four_tok_type_ids, maxlen=MAXLEN, dtype="long", value=0, truncating="post", padding="post")
        input_ids.append(four_inp_ids)
        token_type_ids.append(four_tok_type_ids)

    # Create attention masks
    attention_masks = []
    for sen in input_ids:
        sen_attention_masks = []
        for opt in sen:
            att_mask = [int(token_id > 0) for token_id in opt]
            sen_attention_masks.append(att_mask)
        attention_masks.append(sen_attention_masks)
    # Convert to torch tensors
    input_ids = torch.tensor(np.asarray(input_ids))
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(np.asarray(token_type_ids))
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(np.asarray(attention_masks))
    attention_masks = attention_masks.long().to(device)

    ds = TensorDataset(input_ids, token_type_ids, attention_masks)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    models = []
    seeds = [1, 2, 3]

    for seed in seeds:
        model_path = args.qa_models_dir + "seed" + str(seed) + "/electra_QA_MC_seed" + str(seed) + ".pt"
        model = torch.load(model_path, map_location=device)
        model.eval().to(device)
        models.append(model)

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
        if count==1:
            preds = np.asarray(curr_preds)
        else:
            preds = np.concatenate((preds, curr_preds), axis=1)

    uncs = ensemble_uncertainties_classification(preds)

    for unc_key, curr_uncs in uncs:
        with open(unc_key+'.npy', 'wb') as f:
            np.save(f, curr_uncs)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
