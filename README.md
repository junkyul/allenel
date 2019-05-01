# allenel
* entity linking in allennlp
  * [Original project in TensorFlow by Nitish Gupta](https://nitishgupta.github.io/neural-el/)
  * [Nitish Gupta, Sameer Singh, and Dan Roth, “Entity Linking via Joint Encoding of Types, Descriptions, and Context”, EMNLP 2017](http://cogcomp.org/page/publication_view/817)

## evaluation & demo
* evaluation
```
$ allennlp evaluate <path_to_model.tar.gz> <input test file> --include-package allenel --output-file <output path>
$ allennlp evaluate /home/junkyul/conda/allenel/models/Cmodel_15/model.tar.gz /home/junkyul/conda/neural-el_test/wiki.txt --include-package allenel --outpu-file /home/junkyul/conda/allenel/outputs/wiki.log
```
* demo 
```
$ python -m allennlp.service.server_simple --archive-path ./tests/fixtures/model.tar.gz --predictor el_linker --include-package allenel --title "EL Demo" --field-name context
```

* [google drive link to the archived models](https://drive.google.com/file/d/1RCk84pygWpp2Y0WirlU8Oq3MFyPrgrUE/view?usp=sharing)


