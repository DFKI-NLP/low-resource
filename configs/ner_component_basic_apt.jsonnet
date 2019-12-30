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
  "train_data_path": "src/data/adaptNER/amazon_ner_train_single.jsonl",
  "validation_data_path":"src/data/distantly_labeled/dev_appear_tri.jsonl",
  "evaluate_on_test": false,
  "iterator": {
    "type": "basic",
    "batch_size": 2
  },
  "model": {
    "type": "low_resource_ner_extractor",
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
      "type": "low_resource_crf_tagger",
      "constrain_crf_decoding": true,
      "label_encoding": "BIO",
      "calculate_span_f1": false,
      "dropout": 0.1,
      "include_start_end_transitions": true,
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
    "cuda_device": [4,5,6,7],
    "histogram_interval": 10
  }
}
