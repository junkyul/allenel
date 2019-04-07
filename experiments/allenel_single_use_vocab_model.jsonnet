local n_types = 114;            # len of types vocab
{
  "dataset_reader": {
    "type": "el_reader",
    "resource_path": "/home/junkyul/conda/neural-el_resources",
    "n_types": n_types,
  },
//
//
//  "vocabulary": {
//    "directory_path": "/home/junkyul/conda/allenel/vocab",
//    "extend" : true
//  },
//
//
  "train_data_path": "/home/junkyul/conda/neural-el_train/train.mens.0",
  "validation_data_path": "/home/junkyul/conda/neural-el_test/conll2012_dev.txt",
  "test_data_path": "/home/junkyul/conda/neural-el_test/conll2012_test.txt",
  "evaluate_on_test": true,
//
//
  "model": {
    "type": "el_model",
    "sentence_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "vocab_namespace": "sentences",
          "embedding_dim": 300,
          "sparse": true,
        }
      }
    },
    "entity_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 200,
          "vocab_namespace": "wids",
          "sparse": true
        }
      }
    },
    "coherence_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 100,
          "vocab_namespace": "coherences",
          "sparse": true
        }
      }
    },
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
    "ff_context": {
      "input_dim": 200,
      "num_layers": 1,
      "hidden_dims": 200,
      "activations": "relu",
      "dropout": 0.4
    },
  },
//
  "iterator": {
    "type": "basic",
    "batch_size": 100
  },
//
  "trainer": {
    "num_epochs": 5,
    "cuda_device": [0, 1],    // model device cpu -> -1, gpu -> 0
//    "gradient_norm": 1.0,
    "grad_clipping": 5.0, //if norm fails
//    "validation_metric": "+accuracy",
    "optimizer": {
      //"type": "adam",
      "type": "dense_sparse_adam",
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
