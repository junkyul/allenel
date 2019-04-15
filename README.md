# allenel
* entity linking in allennlp
  * [Original project in TensorFlow by Nitish Gupta](https://nitishgupta.github.io/neural-el/)
  * [Nitish Gupta, Sameer Singh, and Dan Roth, “Entity Linking via Joint Encoding of Types, Descriptions, and Context”, EMNLP 2017](http://cogcomp.org/page/publication_view/817)

## redo project
* demo 
```
$ python -m allennlp.service.server_simple --archive-path ./tests/fixtures/model.tar.gz --predictor el_linker --include-package allenel --title "EL Demo" --field-name context
```