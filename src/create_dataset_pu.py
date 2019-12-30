import sys
import os

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/data2/zhanghc/RE/low-resource'])

from low_resource.data.data_utils import to_iob1_by_spacy

from src.prepare.spilit_train_dev_appear import create_train_dev_split_arne

import json
import gzip
from collections import namedtuple
from typing import List, Dict, Any
from src.pipeline.create_pipline_init import create_nlp_pipeline
from spacy.language import Language
from tqdm import tqdm
import numpy as np

Instance = namedtuple("Instance", ["id", "text"])
AnnotatedInstance = namedtuple("AnnotatedInstance",
                               ["id", "docid", "token", "ner", "pos", "dep_head", "dep_rel", "iob", "ent_num"])


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def instances_from_amazon() -> List[Instance]:
    path = "/data2/zhanghc/data_external/amazon_reviews/meta_Electronics.json.gz"
    i = 0
    instances = []
    for example in parse(path):
        i += 1
        for cat in example['categories'][0]:
            if "Computer" in cat and "description" in example:
                instances.append(Instance(id=example["asin"],
                                          text=example["description"]))
                continue

    np.random.seed(1234)
    np.random.shuffle(instances)
    instances = instances[:int(len(instances) * 0.3)]
    return instances


def instances_from_wdc(path: str) -> List[Dict[str, Any]]:
    examples = []
    with open(path, "r") as dataset_f:
        for line in dataset_f:
            t = json.loads(line)
            examples.append(Instance(id=t["id"],
                                     text=t["text"]))
    return examples


def create_annotated_dataset(instances: List[Instance], nlp: Language, output_path: str) -> None:
    annotated_instances = []
    for instance in instances:
        doc = nlp(instance.text)

        if len(doc.ents) == 0:
            continue

        annotated_instances.append(AnnotatedInstance(id=instance.id,
                                                     docid=instance.id,
                                                     token=[t.text for t in doc],
                                                     ner=[t.ent_type_ if t.ent_type_ else "O" for t in doc],
                                                     pos=[t.pos_ for t in doc],
                                                     iob=[t.ent_iob_ for t in doc],
                                                     dep_head=[t.head.i for t in doc],
                                                     dep_rel=[t.dep_ for t in doc]))

    with open(output_path, "w") as out_f:
        json.dump([instance._asdict() for instance in annotated_instances], out_f)

    return annotated_instances


"""
this method produce the entity with single class
"""


def create_annotated_single_dataset(instances: List[Instance], nlp: Language, output_path: str,
                                    ner_tag="COMPONENTS") -> None:
    annotated_instances = []
    for instance in tqdm(instances):
        doc = nlp(instance.text)

        if len(doc.ents) == 0:
            continue

        annotated_instances.append(AnnotatedInstance(id=instance.id,
                                                     docid=instance.id,
                                                     token=[t.text for t in doc],
                                                     ner=[t.ent_type_ if (
                                                             t.ent_type_ and t.ent_type_ == ner_tag) else "O" for t
                                                          in doc],
                                                     iob=[t.ent_iob_ for t in doc],
                                                     ent_num=len(doc.ents),
                                                     pos=[t.pos_ for t in doc],
                                                     dep_head=[t.head.i for t in doc],
                                                     dep_rel=[t.dep_ for t in doc]))

    with open(output_path, "w") as out_f:
        json.dump([instance._asdict() for instance in annotated_instances], out_f)

    return annotated_instances


def to_binary_label(instance, tag_type, iob_tag=None):
    # encode all tags as "inside" (I-) tags
    for t_idx, t in enumerate(instance[tag_type]):
        if instance[tag_type][t_idx] != 'O':
            instance[tag_type][t_idx] = 'COMPONENT'  # + t
    return instance


def convert_to_jsonl_single_cls(path_in, path_out=None, min_tokens=6, force_recreate=True, isBinaryLabel=False):
    """
    create json line file from json file (has to contain an array of instances)
    We don't need data that much, cost too long to training and overfitting
    """
    print(f'convert to jsonl {path_in} ...')
    if path_out is None:
        path_out = (path_in[:len(path_in) - 5] + '_single.jsonl') if path_in.endswith(
            '.json') else path_in + '_single.jsonl'
    if os.path.exists(path_out):
        if force_recreate:
            print(f'WARNING: {path_out} already exists. File will be overwritten!')
        else:
            print(f'WARNING: {path_out} already exists. File will NOT be recreated!')
            return path_out
    n = 0
    n_discarded = 0
    label_function = to_binary_label if isBinaryLabel else to_iob1_by_spacy
    with open(path_in) as f:
        data = json.load(f)
        with open(path_out, 'w') as out_f:
            for inst in tqdm(data):
                if len(inst['token']) >= min_tokens:
                    out_f.write(json.dumps(label_function(inst, tag_type='ner', iob_tag="iob")) + '\n')
                    n += 1
                else:
                    n_discarded += 1
    print(f'wrote {n} instances to {path_out} (discard {n_discarded} instances because len(tokens) < {min_tokens})')
    return path_out


if __name__ == '__main__':
    GAZETTEERS_PATH = "/data2/zhanghc/RE/low-resource/src/data/gazetteers/"
    nlp = create_nlp_pipeline(gazetteers_path=GAZETTEERS_PATH)
    instances = instances_from_amazon()
    annotated_instances = create_annotated_single_dataset(instances, nlp, output_path="data/amazon_distant.json",ner_tag="COMPONENTS")

    isBinaryLabel = False

    filePath = "data/distantly_labeled/"
    trainName = "train_appear"
    devName = "dev_appear"

    trainName = trainName if isBinaryLabel else trainName + "_tri"
    devName = devName if isBinaryLabel else devName + "_tri"

    path_out = convert_to_jsonl_single_cls(path_in='data/amazon_distant.json', isBinaryLabel=isBinaryLabel)
    create_train_dev_split_arne(path_in=path_out,
                                path_out_train=filePath + trainName + ".jsonl",
                                path_out_dev=filePath + devName + ".jsonl", isBinaryLabel=isBinaryLabel)
    print("path_out=", filePath + trainName + ".jsonl")
