from typing import Dict
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.data import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, TextField, MultiLabelField, MetadataField

from allennlp.common.util import START_SYMBOL, END_SYMBOL

import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("el_reader")
class EnityLinknigDatasetReader(DatasetReader):
    crosswiki: Dict[str, tuple] = {}

    def __init__(self,
                 resource_path: str,
                 ) -> None:
        super().__init__(lazy=True)
        self.reource_path = resource_path
        self.sentence_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter(), start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])
        self.sentence_indexers = {"tokens": SingleIdTokenIndexer(namespace="sentences")}

        self.coherence_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self.coherence_indexers = {"tokens": SingleIdTokenIndexer(namespace="coherences")}

        self.wid_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self.wid_indexers = {"tokens": SingleIdTokenIndexer(namespace="wids")}

        import pickle, os
        logger.info("start reading crosswikis.pruned.pkl")
        EnityLinknigDatasetReader.crosswiki = pickle.load(open(os.path.join(resource_path, "crosswikis.pruned.pkl"), "rb"))
        logger.info("end reading crosswikis.pruned.pkl")

    @overrides
    def _read(self, file_path):
        with open(file_path) as data_file:
            for line in data_file:
                line = line.strip("\n")
                if not line: continue
                temp = line.split("\t")     # separated by Tabs
                free_id, wiki_id, wiki_title = temp[0], temp[1], temp[2]
                st_idx, et_idx, mention_surface, mention_sentence= temp[3], temp[4], temp[5], temp[6]
                free_types = temp[7] if len(temp) > 7 else ""
                coherence_mentions = temp[8] if len(temp) > 8 else ""
                yield self.text_to_instance(wiki_id, wiki_title, int(st_idx.strip()), int(et_idx.strip()),
                                            mention_surface, mention_sentence, free_types, coherence_mentions)

    @staticmethod
    def getLnrm(arg):
        """taken from neural-el project by Nitish. normalizes mention surface and use as a key to crosswiki dict"""
        import unicodedata
        arg = ''.join(
            [c for c in unicodedata.normalize('NFD', arg) if unicodedata.category(c) != 'Mn'])
        arg = arg.lower()
        arg = ''.join(
            [c for c in arg if c in set('abcdefghijklmnopqrstuvwxyz0123456789')])
        return arg

    @overrides
    def text_to_instance(self,
                         wiki_id: str,
                         wiki_title: str,
                         st_idx: int,
                         et_idx: int,
                         mention_surface: str,
                         mention_sentence: str,
                         free_types: str,
                         coherence_mentions: str
                         )-> Instance:

        sentence_tokenized = self.sentence_tokenizer.tokenize(mention_sentence)
        sentence_field = TextField(sentence_tokenized, self.sentence_indexers)
        tokenized_left = [Token(START_SYMBOL)] + sentence_tokenized[1:st_idx+1] + [Token(END_SYMBOL)]
        tokenized_right = [Token(START_SYMBOL)] + sentence_tokenized[-2:et_idx+1:-1] + [Token(END_SYMBOL)]
        sentence_left_field = TextField(tokenized_left, self.sentence_indexers)
        sentence_right_field = TextField(tokenized_right, self.sentence_indexers)       # 3 sentences share vocab

        mention_surface_field = MetadataField(mention_surface)
        mention_surface_normalized = EnityLinknigDatasetReader.getLnrm(mention_surface)
        mention_normalized_field = MetadataField(mention_surface_normalized)
        wid_field = MetadataField(wiki_id)
        title_field = MetadataField(wiki_title)

#        type_field = MultiLabelField(labels=free_types.split(" "), label_namespace="types",
#                                     skip_indexing=False, num_labels=self.n_types)


        coherence_tokenized = self.coherence_tokenizer.tokenize(coherence_mentions)
        coherences_field = TextField(coherence_tokenized, self.coherence_indexers)

        candidates = [wiki_id]
        if mention_surface_normalized in EnityLinknigDatasetReader.crosswiki:
            priors = EnityLinknigDatasetReader.crosswiki[mention_surface_normalized][0]     # probs for prediction
            if wiki_id in priors:
                priors.remove(wiki_id)
            candidates += priors

        candidates_tokenized = [Token(t) for t in candidates]
        if len(candidates_tokenized) == 1:
            candidates_tokenized = candidates_tokenized + [Token("@@UNKNOWN@@")]
        candidates_field = TextField(candidates_tokenized, self.wid_indexers)
        target_field = LabelField(label=0, label_namespace="label", skip_indexing=True)

        fields = {
            "wid": wid_field,                                   # label -> meta
            "title": title_field,                               # label -> meta
 #           "types": type_field,                                # multi label for one hot
            "sentence": sentence_field,                         # text
            "sentence_left": sentence_left_field,               # text
            "sentence_right": sentence_right_field,             # text
            "mention": mention_surface_field,                   # meta
            "mention_normalized": mention_normalized_field,     # meta
            "coherences": coherences_field,                     # multi label -> text
            "candidates": candidates_field,                     # multi label -> text
            "targets": target_field
        }

        return Instance(fields)     # no indexing no vocabulary
