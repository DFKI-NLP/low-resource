{
  local num_gaz_labels = 2,

  local token_emb_dim = 300,
  local char_emb_dim = 16,
  local char_num_filters = 128,

  local hidden_size = 200,
  local num_layers = 2,
  local dropout = 0.3,

  "dataset_reader": {
    "type": "jsonl",
    "tag_label": "ner",
    "coding_scheme": "IOB1",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },
  "train_data_path": "/data2/zhanghc/RE/low-resource/src/data/distantly_labeled/train_appear.jsonl",
  "validation_data_path": "/data2/zhanghc/RE/low-resource/src/data/distantly_labeled/dev_appear.jsonl",
  "evaluate_on_test": false,
  "model": {
    "type": "ner_extractor_pu",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": token_emb_dim,
          "pretrained_file": "/data2/zhanghc/RE/cc.en.300.vec.gz",
          "trainable": false
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
              "embedding_dim": char_emb_dim
          },
          "encoder": {
              "type": "cnn",
              "embedding_dim": char_emb_dim,
              "num_filters": char_num_filters,
              "ngram_filter_sizes": [3],
              "conv_layer_activation": "relu"
          }
        },

      },
    },
    "ner_model": {
      "type": "low_resource_crf_tagger_pu",
      "constrain_crf_decoding": false,
      "label_encoding": "BO",
      "calculate_span_f1": false,
      "dropout": 0.1,
      "prior":0.02 ,
      "gamma": 0.8,
      "m": 0.9,
      "include_start_end_transitions": false,
      "encoder": {
        "type": "lstm",
        "input_size": token_emb_dim + char_num_filters,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": true
      }
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 2
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
    "num_epochs": 1,
    "patience": 5,
    "cuda_device": [4],
    "histogram_interval": 10
  }
}
