#! /usr/bin/env python

"""
The vocab levels are only based on the question + context
"""

import argparse
import os
import sys
import json
import numpy as np

from xml.dom import minidom

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--easy_vocab', type=str, help='vocab')
parser.add_argument('--medium_vocab', type=str, help='vocab')
parser.add_argument('--hard_vocab', type=str, help='vocab')
    

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Process xml files

    easy_list = []
    file = minidom.parse(args.easy_vocab)
    vocab_words = file.getElementsByTagName('di')
    for word in vocab_words:
        easy_list.append(word.attributes['id'].value)

    medium_list = []
    file = minidom.parse(args.medium_vocab)
    vocab_words = file.getElementsByTagName('di')
    for word in vocab_words:
        medium_list.append(word.attributes['id'].value)

    hard_list = []
    file = minidom.parse(args.hard_vocab)
    vocab_words = file.getElementsByTagName('di')
    for word in vocab_words:
        hard_list.append(word.attributes['id'].value)


    with open(args.test_data_path) as f:
        test_data = json.load(f)

    all_words = []

    grades = []
    for item in test_data:
        context = item["context"]
        question = item["question"]
        combo = question + " " + context
        all_words.append(combo)
        grades.append(item['level'])


    vocab_levels = []
    tot_contexts = len(all_words)
    for i, context in enumerate(all_words):
        context = context.lower()
        num_easy = 0.0
        num_medium = 0.0
        num_hard = 0.0
        # extract each word in context separately
        words = context.split()
        for word in words:
            if word in easy_list:
                num_easy+=1
            elif word in medium_list:
                num_medium+=1
            elif word in hard_list:
                num_hard+=1
        vocab_level = (num_easy*0.0 + num_medium*0.5 + num_hard*1.0) / float(num_easy+num_medium+num_hard)
        print("Context num", i+1, "of", tot_contexts, "Easy:", num_easy, "Medium:", num_medium, "Hard:", num_hard)
        vocab_levels.append(vocab_level)
    vocab_levels = np.asarray(vocab_levels)

    grades = np.asarray(grades)
    positions_b1 = np.argwhere(grades=='b1')
    positions_b2 = np.argwhere(grades=='b2')
    positions_c1 = np.argwhere(grades=='c1')
    positions_c2 = np.argwhere(grades=='c2')

    with open('vocab_cupa_b1.npy', 'wb') as f:
        np.save(f, vocab_levels[positions_b1])
    with open('vocab_cupa_b2.npy', 'wb') as f:
        np.save(f, vocab_levels[positions_b2])
    with open('vocab_cupa_c1.npy', 'wb') as f:
        np.save(f, vocab_levels[positions_c1])
    with open('vocab_cupa_c2.npy', 'wb') as f:
        np.save(f, vocab_levels[positions_c2])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
