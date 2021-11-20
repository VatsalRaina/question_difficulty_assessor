#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from datasets import load_dataset
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup


MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=24, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=10, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--train_data_path', type=str, help='Load path of training data')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')

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

    with open(args.train_data_path + "middle.json") as f:
        middle_data = json.load(f)
    with open(args.train_data_path + "high.json") as f:
        high_data = json.load(f)
    with open(args.train_data_path + "college.json") as f:
        college_data = json.load(f)
    train_data = middle_data + high_data + college_data

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

    targets = [0]*num_middle + [1]*num_high + [2]*num_college

    input_ids = []
    token_type_ids = []
    attention_masks = []

    for item in train_data:
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

    train_data = TensorDataset(input_ids, token_type_ids, attention_masks, targets)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model = ElectraForSequenceClassification.from_pretrained(
        electra_large, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 3, # Perform regression.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    model.to(device)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    # weight_decay = 0.01
                    )

    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 306,
                                                num_training_steps = total_steps)


    for epoch in range(args.n_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_tok_typ_ids = batch[1].to(device)
            b_att_msks = batch[2].to(device)
            b_targets = batch[3].to(device)
            model.zero_grad()

            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, token_type_ids=b_tok_typ_ids, labels=b_targets)
            loss = outputs[0]
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # Save the model to a file
    file_path = args.save_path+'electra_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
