from allennlp.commands import main

# This script is for debugging purposes.
#
# If no debugging is required, execute e.g. this command from the command line to train with a basic NER model
# (see model config "config/ner_basic.jsonnet") and save model checkpoints in "checkpoints/debug"
# (ATTENTION: "-f" indicates deletion of all content previously created in "checkpoints/debug"!):
#   allennlp train -s checkpoints/debug -f configs/ner_basic.jsonnet --include-package low_resource

# E.g. call this script via:
#   python debug_train.py train -s checkpoints/debug -f configs/ner_basic.jsonnet --include-package low_resource


# python debug_train.py evaluate --include-package low_resource --cuda-device 0  checkpoints/debug src/data/distantly_labeled/dev_appear.jsonl
# python debug_train.py predict --output-file src/outputs/predict_spacy.jsonl --cuda-device 1  --predictor low-resource-tagger-predictor  --include-package low_resource checkpoints/debug src/data/distantly_labeled/dev_appear.jsonl

if __name__ == '__main__':
    main()
