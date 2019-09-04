#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 04.09.19
@author: leonhard.hennig@dfki.de
"""
import json
from random import shuffle
import numpy as np


def to_iob1(instance):
    for t_idx in range(len(instance['ner'])):
        if instance['ner'][t_idx] != 'O':
            instance['ner'][t_idx] = 'I-' + instance['ner'][t_idx]
    for t_idx in range(len(instance['pos'])):
        if instance['pos'][t_idx] != 'O':
            instance['pos'][t_idx] = 'I-' + instance['pos'][t_idx]
    return instance


def convert_json_to_jsonl(path, train, dev):
    '''
    hacky one-off conversion script to create two jsonl files from train_distant.json
    '''
    with open(path) as f:
        data = json.load(f)
        indices = np.random.permutation(len(data))
        with open(train, 'w') as out_train:
            for idx in indices[:int(np.round(len(data) * 0.8))]:
                if len(data[idx]['token']) > 5:
                    out_train.write(json.dumps(to_iob1(data[idx])) + '\n')
        with open(dev, 'w') as out_dev:
            for idx in indices[int(np.round(len(data) * 0.8)):]:
                if len(data[idx]['token']) > 5:
                    out_dev.write(json.dumps(to_iob1(data[idx])) + '\n')

if __name__ == '__main__':
    convert_json_to_jsonl('../../data/train_distant.json', '../../data/train_distant.jsonl', '../../data/dev_distant.jsonl')