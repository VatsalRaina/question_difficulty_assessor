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
import matplotlib.pyplot as plt

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=24, help='Specify the training batch size')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--predictions_save_path', type=str, help='Load path to which predictions will be saved')


latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
	assert(len(text_list) == len(attention_list))
	if rescale_value:
		attention_list = rescale(attention_list)
	word_num = len(text_list)
	text_list = clean_word(text_list)
	with open(latex_file,'w') as f:
		f.write(r'''\documentclass[varwidth]{standalone}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
		string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{\tiny'''+"\n"
		for idx in range(word_num):
			string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
		string += "\n}}}"
		f.write(string+'\n')
		f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list

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


    targets = [0]*num_middle + [1]*num_high + [2]*num_college
    targets = np.asarray(targets)

    input_ids = []
    token_type_ids = []
    attention_masks = []

    all_combos = []

    for item in test_data:
        context = item["article"]
        questions = item["questions"]
        for question in questions:
            combo = question + " [SEP] " + context
            all_combos.append(combo)
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

    print(input_ids.size())

    ds = TensorDataset(input_ids, token_type_ids, attention_masks)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    count = 0
    for inp_id, tok_typ_id, att_msk in dl:
        print(count)
        count+=1
        if count < 300:
            continue
        model.zero_grad()
        inp_id, tok_typ_id, att_msk = inp_id.to(device), tok_typ_id.to(device), att_msk.to(device)
        embedding_matrix = model.electra.embeddings.word_embeddings
        b_inputs_embeds = torch.tensor(embedding_matrix(inp_id), requires_grad=True)
        outputs = model(inputs_embeds=b_inputs_embeds, attention_mask=att_msk, token_type_ids=tok_typ_id)
        curr_pred = torch.sum(torch.squeeze(outputs[0]))
        curr_pred.backward()
        saliency_scores = torch.squeeze(torch.norm(b_inputs_embeds.grad.data.abs(), dim=-1)).detach().cpu().numpy()

        if count == 300:
            break

    # get rid of the first [CLS] token
    saliency_scores = saliency_scores[1:]
    words = tokenizer.tokenize(all_combos[count-1])
    if len(words) > (MAXLEN - 2):
        words = words[:MAXLEN-2]
    print(words)
    saliency_scores_all = saliency_scores
    print(saliency_scores_all)
    saliency_scores = saliency_scores[:len(words)]
    print(len(words), saliency_scores)
    

    M = len(words)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(40,20))
    plt.barh(xx, list(saliency_scores)[::-1], color="blue")
    plt.yticks(xx, labels=np.flip(words), fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Input')
    plt.ylim([-2, M+2])
    # plt.xlim([0.0, 0.17])
    plt.savefig('./saliency.png')
    plt.clf()

    # Create textual heatmap
    cleaned_words = []
    for word in words:
        cleaned_words.append(word.replace('#', ''))
    generate(cleaned_words, saliency_scores, "example.tex", 'red', True)




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
