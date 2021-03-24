// Configuration for a named entity recognization model based on:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).
//   taken from https://gist.github.com/joelgrus/7cdb8fb2d81483a8d9ca121d9c617514 (slightly modified)

{
  local data_dir = "/home/arne/data/datasets/ner/conll2003",
  //local requires_grad = std.extVar("BERT_FINETUNE"),

  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "tokens": {
         "type": "bert-pretrained",
          "pretrained_model": "bert-base-cased",
          "do_lowercase": false,
          "use_starting_offsets": true
      }
    }
  },
  "train_data_path": data_dir + "/train_extract10.txt", //std.extVar("CONLL2003_TRAIN_DATA_PATH"),
  "validation_data_path": data_dir + "/valid_extract10.txt", //std.extVar("CONLL2003_VAL_DATA_PATH"),
  //"test_data_path": std.extVar("CONLL2003_TEST_DATA_PATH"),
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
          "tokens": ["tokens", "tokens-offsets"],
        },
        "token_embedders": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-cased",
                //"requires_grad": requires_grad,
            }
        }
    },
    "encoder": {
        "type": "lstm",
        "input_size": 768,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 128
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    //"num_serialized_models_to_keep": 3,
    "checkpointer": {
       "num_serialized_models_to_keep": 3
    },
   "num_epochs": 30,
    "grad_norm": 5.0,
    "patience": 25,
    //"cuda_device": std.map(std.parseInt, std.split(std.extVar("CUDA_VISIBLE_DEVICES"), ","))
    "cuda_device": 0 //std.range(0, std.length(std.split(std.extVar("CUDA_VISIBLE_DEVICES"), ",")) - 1)
  }
}