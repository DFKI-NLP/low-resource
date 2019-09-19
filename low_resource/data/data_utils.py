#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 04.09.19
@author: leonhard.hennig@dfki.de
"""
import json
import os

import numpy as np


def to_iob1(instance, tag_type):
    # encode all tags as "inside" (I-) tags
    last_t = None
    for t_idx, t in enumerate(instance[tag_type]):
        if instance[tag_type][t_idx] != 'O':
            if t != last_t:
                instance[tag_type][t_idx] = 'B-' + t
            else:
                instance[tag_type][t_idx] = 'I-' + t
        last_t = t
    #for t_idx in range(len(instance['pos'])):
    #    if instance['pos'][t_idx] != 'O':
    #        instance['pos'][t_idx] = 'I-' + instance['pos'][t_idx]
    return instance


def convert_json_to_jsonl_bio(path_in, path_out=None, min_tokens=6, force_recreate=False):
    """
    create json line file from json file (has to contain an array of instances)
    """
    print(f'convert to jsonl {path_in} ...')
    if path_out is None:
        path_out = (path_in[:len(path_in)-5] + '_bio.jsonl') if path_in.endswith('.json') else path_in + '_bio.jsonl'
    if os.path.exists(path_out):
        if force_recreate:
            print(f'WARNING: {path_out} already exists. File will be overwritten!')
        else:
            print(f'WARNING: {path_out} already exists. File will NOT be recreated!')
            return path_out
    n = 0
    n_discarded = 0
    with open(path_in) as f:
        data = json.load(f)
        with open(path_out, 'w') as out_f:
            for inst in data:
                if len(inst['token']) >= min_tokens:
                    out_f.write(json.dumps(to_iob1(inst, tag_type='ner')) + '\n')
                    n += 1
                else:
                    n_discarded += 1
    print(f'wrote {n} instances to {path_out} (discard {n_discarded} instances because len(tokens) < {min_tokens})')
    return path_out


def create_train_dev_split(path, train, dev, proportion_train=0.8):
    with open(path) as f:
        lines = f.readlines()
    indices = np.random.permutation(len(lines))
    k = int(np.round(len(lines) * proportion_train))
    open(train, 'w').writelines((lines[idx] for idx in indices[:k]))
    print(f'created {train} ({k} instances)')
    open(dev, 'w').writelines((lines[idx] for idx in indices[k:]))
    print(f'created {dev} ({len(indices) - k} instances)')


if __name__ == '__main__':
    path_out = convert_json_to_jsonl_bio('../../data/distantly_labeled/amazon_wdc_washer_distant.json')
    #path_out = convert_json_to_jsonl('../../data/distantly_labeled/lenovo_distant.json')
    create_train_dev_split(path_out, '../../data/train_distant.jsonl', '../../data/dev_distant.jsonl')