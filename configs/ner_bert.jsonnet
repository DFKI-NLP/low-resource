{
  local num_gaz_labels = 3,

  local token_emb_dim = 768,
  local char_emb_dim = 16,
  local char_num_filters = 128,

  local hidden_size = 200,
  local num_layers = 2,
  local dropout = 0.3,

  "dataset_reader": {
    "type": "jsonl",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        "do_lowercase": false,
        "use_starting_offsets": true,
        "truncate_long_sequences": false,
      },
      // "token_characters": {
      //   "type": "characters",
      //   "min_padding_length": 3
      // },
      "ner_tokens": {
        "type": "ner_tag"
      },
    }
  },
  "train_data_path": "data/train_distant.jsonl",
  "validation_data_path": "data/dev_distant.jsonl",
  "evaluate_on_test": false,
  "model": {
    "type": "low_resource_ner_extractor",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "tokens": ["tokens", "tokens-offsets"],
        "ner_tokens": ["ner_tokens"],
      },
      "token_embedders": {
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
        },
        // "token_characters": {
        //   "type": "character_encoding",
        //   "embedding": {
        //       "embedding_dim": char_emb_dim
        //   },
        //   "encoder": {
        //       "type": "cnn",
        //       "embedding_dim": char_emb_dim,
        //       "num_filters": char_num_filters,
        //       "ngram_filter_sizes": [3],
        //       "conv_layer_activation": "relu"
        //   }
        // },
        "ner_tokens": {
          "type": "one_hot",
        },
      },
    },
    "ner_model": {
      "type": "low_resource_crf_tagger",
      "constrain_crf_decoding": true,
      "label_encoding": "BIOUL",
      "calculate_span_f1": true,
      "dropout": 0.1,
      "include_start_end_transitions": true,
      "encoder": {
        "type": "lstm",
        "input_size": token_emb_dim + num_gaz_labels,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": true
      },
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8
  },
  "trainer": {
    "optimizer": {
      "type": "adam",
      "lr": 1e-3
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "validation_metric": "+ner-f1-measure-overall",
    "num_serialized_models_to_keep": 1,
    "grad_clipping": 5.0,
    "num_epochs": 20,
    "patience": 5,
    "cuda_device": -1
  }
}
