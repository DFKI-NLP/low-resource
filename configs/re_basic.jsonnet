{
  local token_emb_dim = 300,
  local offset_emb_dim = 50,
  local max_offset = 100,

  local emb_dropout = 0.5,
  local encoding_dropout = 0.5,

  local num_filters = 150,

  "dataset_reader": {
    "type": "semeval_2010_task_8",
    "max_len": 100,
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
        19
      ],
      "input_dim": 600,
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
      "embedding_dim": 400,
      "ngram_filter_sizes": [
        2,
        3,
        4,
        5
      ],
      "num_filters": num_filters
    },
    "text_field_embedder": {
      "tokens": {
        "type": "embedding",
        "embedding_dim": token_emb_dim,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
        "trainable": false
      }
    },
    "verbose_metrics": false,
  },
  "train_data_path": "data/semeval_2010_task_8/train.jsonl",
  "validation_data_path": "data/semeval_2010_task_8/dev.jsonl",
  "trainer": {
    "cuda_device": 0,
    "num_epochs": 50,
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
