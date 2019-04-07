local n_types = 114;            # len of types vocab
//local n_coherences = 3753616;   # len of coherences vocab
//local n_entities = 408627;      # len of wid vocab
{
  "dataset_reader": {
    "type": "el_reader",
    "resource_path": "/home/junkyul/conda/neural-el_resources",
    "n_types": n_types,
//    "n_coherences": n_coherences
  },
//
//
//  "vocabulary": {
//    "directory_path": "/home/junkyul/conda/allenel/vocab",
//    "extend" : true
//  },
//
//
//  "train_data_path": "/home/junkyul/conda/neural-el_train/train_short.mens",
  "train_data_path": "/home/junkyul/conda/neural-el_train/train.mens.0",
  "validation_data_path": "/home/junkyul/conda/neural-el_test/conll2012_dev_short.txt",
  "test_data_path": "/home/junkyul/conda/neural-el_test/conll2012_test_short.txt",
  "evaluate_on_test": true,
//
//
  "model": {
    "type": "el_model",
//
//"text_field_embedder": {
//    "tokens": {
//                "type": "embedding",
//                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
//                "embedding_dim": 100,
//                "trainable": false
//     }
//},
//    "sentence_embedder": {
//      "token_embedder": {
//        "tokens": {
//          "type": "embedding",
//          "embedding_dim": 300,
////          "pretrained_file": "/home/junkyul/conda/glove/glove.840B.300d.txt.gz",
//        }
//      },
//    },
//
    "sentence_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "sentences",
          "embedding_dim": 300
        }
      }
    },
    "entity_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 200,
//          "padding_index": 0,
          "vocab_namespace": "wids"
        }
      }
//      "wids": {
//        "type": "embedding",
//        "embedding_dim": 200,
//        "vocab_namespace": "wids",
//      }
    },
//
    "coherence_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,
//          "padding_index": 0,
          "vocab_namespace": "coherences"
        }
      }
//      "coherences": {
//        "type": "embedding",
//        "embedding_dim": 100,
//        "vocab_namespace": "coherences",
//      }
    },
//
    "left_seq2vec": {
      "type": "lstm",
        "bidirectional": false,
        "input_size": 300,
        "hidden_size": 100,
        "num_layers": 1
    },
//
    "right_seq2vec": {
      "type": "lstm",
        "bidirectional": false,
        "input_size": 300,
        "hidden_size": 100,
        "num_layers": 1
    },
//
    "ff_seq2vecs" : {
      "input_dim": 200,
      "num_layers": 1,
      "hidden_dims": 100,
      "activations": "relu",
      "dropout": 0.4
    },
//
//    "coherence_embedding_opt": {
//      "num_embeddings": n_coherences,
//      "embedding_dim": 100,
//      "padding_index" : 0,
//      "trainable": true,
//      "vocab_namespace": "coherences",
//      "sparse": false
//    },
//
    "ff_context": {
      "input_dim": 200,
      "num_layers": 1,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.4
    },
//
//    "entity_embedding_opt": {
//      "num_embeddings": n_entities,
//      "embedding_dim": 200,
//      "padding_index" : 0,
//      "trainable": true,
//      "vocab_namespace": "wid",
//      "sparse": false
//    },
  },
//
//
  "iterator": {
    "type": "basic",
    "batch_size": 2
//    "max_instances_in_memory": 1000,
//    "cache_instances": true
  },
//
//
  "trainer": {
    "num_epochs": 50,
    "cuda_device": 0,    // model device cpu -> -1, gpu -> 0
//    "gradient_norm": 1.0,
    "grad_clipping": 5.0, //if norm fails
//    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam",
//      "type": "dense_sparse_adam",
      "lr": 0.005
    },
    "patience": 10
  }
}

//The available optimizers are
//
//“adadelta”
//“adagrad”
//“adam”
//“sparse_adam”
//“sgd”
//“rmsprop
//“adamax
//“averaged_sgd
