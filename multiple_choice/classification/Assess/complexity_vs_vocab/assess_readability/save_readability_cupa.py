#! /usr/bin/env python

"""
The vocab levels are only based on the question + context
"""

import argparse
import os
import sys
import json
import numpy as np

from readability import Readability

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
    

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')



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


    all_scores = {
        'flesch_kincaid': {'b1':[], 'b2':[], 'c1':[], 'c2':[]}, 
        'flesch': {'b1':[], 'b2':[], 'c1':[], 'c2':[]},
        'gunning_fog': {'b1':[], 'b2':[], 'c1':[], 'c2':[]},
        'coleman_liau': {'b1':[], 'b2':[], 'c1':[], 'c2':[]},
        'dale_chall': {'b1':[], 'b2':[], 'c1':[], 'c2':[]},
        'ari': {'b1':[], 'b2':[], 'c1':[], 'c2':[]},
        'linsear_write': {'b1':[], 'b2':[], 'c1':[], 'c2':[]},
        'smog': {'b1':[], 'b2':[], 'c1':[], 'c2':[]},
        'spache': {'b1':[], 'b2':[], 'c1':[], 'c2':[]}
        }
    for grade, text in zip(grades, all_words):
        text = text.lower()
        r = Readability(text)
        all_scores['flesch_kincaid'][grade].append(r.flesch_kincaid().score)
        all_scores['flesch'][grade].append(r.flesh().score)
        all_scores['gunning_fog'][grade].append(r.gunning_fog().score)
        all_scores['coleman_liau'][grade].append(r.coleman_liau().score)
        all_scores['dale_chall'][grade].append(r.dale_chall().score)
        all_scores['ari'][grade].append(r.ari().score)
        all_scores['linsear_write'][grade].append(r.linsear_write().score)
        all_scores['smog'][grade].append(r.smog().score)
        all_scores['spache'][grade].append(r.spache().score)

    for measure in all_scores.keys():
        print("**************")
        print("Measure:", measure)
        for grade in all_scores[measure].keys():
            curr_scores = np.asarray(all_scores[measure][grade])
            print(np.mean(curr_scores), np.std(curr_scores))



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
