import json


##These data are statistically derived from the annotated data.
tags_list = [('microphone', 5279), ('Hard Drive', 5589), ('Processor', 6073), ('speaker', 6293), ('Mouse', 6504),
             ('Display', 7164), ('multimedia', 7239), ('Laptop Battery', 7467), ('AC Adapter', 7728),
             ('speakers', 7949), ('AMD', 10817), ('Keyboard', 11345), ('mouse', 11514), ('Bluetooth', 11591),
             ('desktop', 12207), ('hard drive', 12572), ('RAM', 12653), ('Screen', 13749), ('processor', 15456),
             ('Battery', 16345), ('display', 18724), ('Memory', 19868), ('keyboard', 23267), ('DVD', 25897),
             ('Laptop', 26509), ('battery', 42657), ('memory', 49813), ('screen', 63832), ('laptop', 69939),
             ('USB', 107291)]
# I use these four tag to split the dataset.
# If a document has one of the tag, it will be add to "dev". otherwise will be "training".
tags_set = {'USB','Keyboard','Memory','RAM'}


def create_train_dev_split(path_in, path_out_train, path_out_dev):
    train_count = 0
    dev_count = 0

    train_B = 1
    dev_B = 1
    with open(path_in) as f, open(path_out_train, 'w') as out_f_train, open(path_out_dev, 'w') as out_f_dev:
        lines = f.readlines()
        for line in lines:
            json_data = json.loads(line)
            tokens = json_data['token']
            ners = json_data['ner']

            seq_len = len(tokens)
            for i in range(len(tokens)):
                tag = ners[i]
                if tag != "B-COMPONENTS":
                    continue

                j = i + 1
                while j < seq_len:
                    if ners[j] != 'I-COMPONENTS':
                        break
                    j += 1
                component = " ".join(tokens[i:j])
                if component in tags_set:
                    out_f_dev.write(line)
                    dev_B += ners.count("B-COMPONENTS")
                    dev_count += 1
                    break
                else:

                    out_f_train.write(line)
                    train_count += 1
                    train_B += ners.count("B-COMPONENTS")
                    break
    print(train_count)
    print(dev_count)
    print("per train doc has: "+str(train_B/train_count))
    print("per dev doc has: "+str(dev_B / dev_count))


def create_train_dev_split_arne(path_in, path_out_train, path_out_dev):
    train_count = 0
    dev_count = 0
    with open(path_in) as f, open(path_out_train, 'w') as out_f_train, open(path_out_dev, 'w') as out_f_dev:
        for line in f.readlines():
            json_data = json.loads(line)
            tokens = json_data['token']
            ners = json_data['ner']

            # get all tokens tagged as component
            # normalize to lower
            tokens_component = [tokens[i].lower() for i, ner in enumerate(ners) if ner in ['B-COMPONENTS', 'I-COMPONENTS']]
            # test if any token in our selected dev component tokens is still there
            tags_set_found = any([t.lower() in tokens_component for t in tags_set])
            if tags_set_found:
                out_f_dev.write(line)
                dev_count += 1
            else:
                out_f_train.write(line)
                train_count += 1
    print(train_count)
    print(dev_count)


create_train_dev_split(path_in="../../src/data/amazon_distant_bio.jsonl",path_out_train="../../src/data/distantly_labeled/train_appear.jsonl",path_out_dev="../../src/data/distantly_labeled/dev_appear.jsonl")

#if __name__ == '__main__':
#    create_train_dev_split_arne(path_in="data/distantly_labeled/amazon_distant_bio.jsonl",
#                           path_out_train="data/distantly_labeled/train_appear.jsonl",
#                           path_out_dev="data/distantly_labeled/dev_appear.jsonl")
