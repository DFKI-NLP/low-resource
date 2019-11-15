{
  local token_emb_dim = 300,
  local offset_emb_dim = 50,
  local max_offset = 150,

  local emb_dropout = 0.5,
  local encoding_dropout = 0.5,

  local num_filters = 150,
  local ngram_filter_sizes = [2, 3],
  local padding_length = 5,

  local text_encoder_input_dim = token_emb_dim + 2 * offset_emb_dim,
  local classifier_feedforward_input_dim = num_filters * std.length(ngram_filter_sizes),

  local num_classes = 6,

  "dataset_reader": {
    "type": "pc_relation_reader",
    "max_len": 150,
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      }
    }
  },
  "iterator": {
    "type": "bucket",
    "batch_size": 20,
    "sorting_keys": [
      [
        "text",
        "num_tokens"
      ]
    ]
  },
  "model": {
    "type": "low_resource_relation_classifier",
    "classifier_feedforward": {
      "activations": [
        "linear"
      ],
      "dropout": [
        0
      ],
      "hidden_dims": [
        num_classes
      ],
      "input_dim": classifier_feedforward_input_dim,
      "num_layers": 1
    },
    "embedding_dropout": emb_dropout,
    "encoding_dropout": encoding_dropout,
    "f1_average": "macro",
    "offset_embedder_head": {
      "type": "relative",
      "embedding_dim": offset_emb_dim,
      "n_position": max_offset
    },
    "offset_embedder_tail": {
      "type": "relative",
      "embedding_dim": offset_emb_dim,
      "n_position": max_offset
    },
    "regularizer": [
      [
        "text_encoder.conv_layer_.*weight",
        {
          "alpha": 1e-05,
          "type": "l2"
        }
      ]
    ],
    "text_encoder": {
      "type": "cnn",
      "embedding_dim": text_encoder_input_dim,
      "ngram_filter_sizes": ngram_filter_sizes,
      "num_filters": num_filters
    },
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": token_emb_dim,
        "pretrained_file": "../glove.840B.300d.txt.gz",
        "trainable": false
      }
    },
    "verbose_metrics": true,
  },
  "train_data_path": "../src/data/distantly_labeled/relation_annotation_train.csv",
  "validation_data_path": "../src/data/distantly_labeled/relation_annotation_dev.csv",
  "trainer": {
    "cuda_device":[1,2,3,4],
    "num_epochs": 3,
    "num_serialized_models_to_keep": 1,
    "optimizer": {
      "type": "adadelta",
      "eps": 1e-06,
      "lr": 1,
      "rho": 0.9
    },
    "validation_metric": "+accuracy"
  },
  "vocabulary": {
    "min_count": {
      "tokens": 2
    }
  }
}
