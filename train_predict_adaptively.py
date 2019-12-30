import os
import sys
import logging

APT_NEW_ADD_ENT_LOG_TXT = "src/data/gazetteers_apt/new_add_ent_log.txt"

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/data2/zhanghc/RE/low-resource'])

from subprocess import call
import json
from typing import List, Dict, Any
from collections import namedtuple
import shutil

from src.create_dataset_pu import create_annotated_single_dataset, convert_to_jsonl_single_cls
from src.pipeline.create_pipline_init import create_nlp_pipeline

Instance = namedtuple("Instance", ["id", "text"])

checkpoints_path = "checkpoints/debug_apt"
config_file = "configs/ner_component_basic_apt.jsonnet"
GAZETTEERS_PATH = "src/data/gazetteers_apt/"
GAZ_PATH = GAZETTEERS_PATH+"components.gaz"

logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("/data2/zhanghc/RE/low-resource/src/data/gazetteers_apt/spam.log")
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)
logger.addHandler(fh)

def read_gazzeteer():
    with open("src/data/gazetteers_apt/components.gaz") as f:
        ents = [line.lower().strip() for line in f]
        return ents


def collect_prediction():
    dict_data = {}
    with open("src/outputs/predict_new_ent.jsonl") as f:
        for line in f:
            data = json.loads(line)
            ents = data["ents"]
            for ent in ents:
                component = ent["name"].lower().strip()
                if component in dict_data:
                    dict_data[component] += 1
                else:
                    dict_data[component] = 1
    return dict_data


def read_corpus_from_previous(path_in) -> List[Instance]:
    instances = []
    with open(path_in) as f:
        for line in f.readlines():
            json_data = json.loads(line)
            tokens = json_data['token']
            instances.append(Instance(id=json_data["id"],
                                      text=" ".join(tokens)))
    return instances


def main():
    # train_comman()
    # if os.path.exists(APT_NEW_ADD_ENT_LOG_TXT):
    #     rm_log_command = "rm  "+APT_NEW_ADD_ENT_LOG_TXT
    #     logger.info(rm_log_command)
    #     call(rm_log_command, shell=True)
    #
    # shutil.copyfile('/data2/zhanghc/RE/low-resource/src/data/gazetteers/components.gaz', '/data2/zhanghc/RE/low-resource/src/data/gazetteers_apt/components.gaz')

    try:
        big_loop_num = 3
        while big_loop_num < 10:
            num_new_ent = predict_new_entity(big_loop_num)
            if num_new_ent == 0:
                print("====finish here====")
                break
            logger.info("{}th start label corpus".format(big_loop_num))
            print("{}th start label corpus".format(big_loop_num))
            #input(" a new loop , please confirm to continue....")

            nlp = create_nlp_pipeline(gazetteers_path=GAZETTEERS_PATH)
            train_json = 'src/data/adaptNER/amazon_ner_train.json'
            dev_json = "src/data/adaptNER/amazon_ner_dev.json"

            instances = read_corpus_from_previous("src/data/distantly_labeled/train_appear_tri.jsonl")
            create_annotated_single_dataset(instances, nlp, output_path=train_json, ner_tag="COMPONENTS")
            tarin_jsonl_path = convert_to_jsonl_single_cls(path_in=train_json, isBinaryLabel=False)

            ##"src/data/adaptNER/amazon_ner_dev_single.jsonl",iza/k m          g gdce  vvvvvvvvv
            #instances = read_corpus_from_previous("src/data/distantly_labeled/dev_appear_tri.jsonl")
            #create_annotated_single_dataset(instances, nlp, output_path=dev_json, ner_tag="COMPONENTS")
            #dev_jsonl_path = convert_to_jsonl_single_cls(path_in=dev_json, isBinaryLabel=False)

            num_path_fine_tune = checkpoints_path + "_" + str(big_loop_num)
            fine_tune_command = "allennlp fine-tune -m {} -c {} -s {}  --include-package low_resource".format(
                checkpoints_path + "_" + str(big_loop_num - 1), config_file,
                num_path_fine_tune)

            if os.path.exists(num_path_fine_tune):
                rm_log_command ="rm -r "+num_path_fine_tune
                call(rm_log_command, shell=True)

            logger.info("excute command: {}".format(fine_tune_command))
            #input("please confirm to continue....")
            call(fine_tune_command, shell=True)

            big_loop_num += 1
    except:
        logger.info("error!!!!!!!!!!!")
        return


def predict_new_entity(loop_num):
    logger.info("finished train,start prediction")
    prediction_command = "python debug_train.py predict --output-file src/outputs/predict_new_ent.jsonl --cuda-device 4  --predictor low-resource-tagger-predictor-apt  --include-package low_resource {}  src/data/distantly_labeled/dev_appear_tri.jsonl".format(
        checkpoints_path + "_" + str(loop_num-1))
    logger.info(prediction_command)
    call(prediction_command, shell=True)
    logger.info("finish prediction , start to manage prediction result")
    dict_data = collect_prediction()
    ents_gaz_set = set(read_gazzeteer())
    new_add_ent_dic = {}
    for ent, value in dict_data.items():
        if ent not in ents_gaz_set and value > 3:
            ents_gaz_set.add(ent)
            new_add_ent_dic[ent] = value
    if len(new_add_ent_dic) > 0:
        # write down the newly added entity and wirte down log
        with open(GAZ_PATH, 'w') as out_f, open(
                APT_NEW_ADD_ENT_LOG_TXT, 'a+') as out_f_log:
            out_f_log.write("=== prediction from debug_apt_{} ===\n".format(str(loop_num-1)))
            out_f_log.write("=== number of new entities {} ===\n".format(str(len(new_add_ent_dic))))
            for ent in ents_gaz_set:
                out_f.write(ent + '\n')
            for ent, value in new_add_ent_dic.items():
                out_f_log.write(ent + " " + str(value) + "\n")
        logger.info("{}th loop output {}".format(loop_num-1, len(new_add_ent_dic)))
        return len(new_add_ent_dic)
    else:
        # we should stop here
        logger.info("now we finished")
        return 0


def train_comman():
    # train_command = train_comman(checkpoints_path, config_file)
    logger.info("we start to train")
    train_command = "python debug_train.py train -s {} -f {} --include-package low_resource".format(
        checkpoints_path + "_0",
        "configs/ner_component_basic.jsonnet")
    logger.info(train_command)
    call(train_command, shell=True)
    return train_command


if __name__ == '__main__':
    main()

# override_config = json.dumps({
#     "train_data_path": tarin_jsonl_path,
#     "validation_data_path": dev_jsonl_path,
# })