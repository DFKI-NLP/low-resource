import json
import random



import numpy as np

def create_train_dev_split(path_in, path_out_train, path_out_dev):
    proportion_train = 0.8


    with open(path_in) as f, open(path_out_train, 'w') as out_f_train, open(path_out_dev, 'w') as out_f_dev:
        lines = f.readlines()

        # given a proportion, I split the dataset to train/dev
        k = int(np.round(len(lines) * proportion_train))

        train_data = lines[:k]
        dev_data = lines[k:]

        out_f_dev.writelines(dev_data) # I just write down all the dev dataset.

        for line in train_data:
            json_data = json.loads(line)
            tokens = json_data['token']
            ners = json_data['ner']

            seq_len = len(tokens)
            for i in range(len(tokens)):
                tag = ners[i]
                if tag != "B-COMPONENTS":
                    continue

                ## if flag is set to TRUE, then we mask this tokens
                flag = True if random.randint(0, 9) < 2 else False
                if flag:
                    tokens[i] = "[BLANK]"
                    j = i + 1
                    while j < seq_len:
                        if ners[j] != 'I-COMPONENTS':
                            break
                        else:
                            tokens[j] = "[BLANK]"
                        j += 1

            out_f_train.write(json.dumps(json_data)+"\n")




create_train_dev_split(path_in="../../src/data/amazon_distant_bio.jsonl",path_out_train="../../src/data/distantly_labeled/train_mask.jsonl",path_out_dev="../../src/data/distantly_labeled/dev_mask.jsonl")
