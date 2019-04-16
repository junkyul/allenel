from typing import Dict
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.data import Token
from allennlp.data.instance import Instance
from allennlp.data.fields import LabelField, TextField, MultiLabelField, MetadataField, ListField, ArrayField

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allenel.dataset_readers.el_multiproc_reader import EnityLinknigDatasetMultiReader

import numpy as np
import logging
logger = logging.getLogger(__name__)


@DatasetReader.register("el_reader")
class EnityLinknigDatasetReader(DatasetReader):
    crosswiki: Dict[str, tuple] = {}
    def __init__(self,
                 resource_path: str = "",
                 ) -> None:
        super().__init__(lazy=True)
        self.reource_path = resource_path
        self.sentence_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter(), start_tokens=[START_SYMBOL], end_tokens=[END_SYMBOL])
        self.sentence_indexers = {"tokens": SingleIdTokenIndexer(namespace="sentences")}

        self.coherence_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self.coherence_indexers = {"tokens": SingleIdTokenIndexer(namespace="coherences")}

        self.wid_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self.wid_indexers = {"tokens": SingleIdTokenIndexer(namespace="wids")}

        self.type_tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())
        self.type_indexers = {"tokens": SingleIdTokenIndexer(namespace="types")}

        if resource_path:
            import pickle, os
            logger.info("start reading crosswikis.pruned.pkl")
            EnityLinknigDatasetReader.crosswiki = pickle.load(open(os.path.join(resource_path, "crosswikis.pruned.pkl"), "rb"))
            logger.info("end reading crosswikis.pruned.pkl")
        elif EnityLinknigDatasetMultiReader.crosswiki:
                EnityLinknigDatasetReader.crosswiki = EnityLinknigDatasetMultiReader.crosswiki

    @overrides
    def _read(self, file_path):
        with open(file_path) as data_file:
            for line in data_file:
                yield self.text_to_instance(*self.process_line(line))

    def process_line(self, line):
        line = line.strip("\n")
        temp = line.split("\t")  # separated by Tabs
        wiki_id = temp[1]
        st_idx, et_idx, mention_surface, mention_sentence = temp[3], temp[4], temp[5], temp[6]
        free_types = temp[7]  # if len(temp) > 7 else "@@UNKNOWN@@"
        coherence_mentions = temp[8] #if len(temp) > 8 else "@@UNKNOWN@@"
        return int(st_idx.strip()), int(et_idx.strip()), mention_surface, mention_sentence, free_types, coherence_mentions, wiki_id

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
                         st_idx: int,
                         et_idx: int,
                         mention_surface: str,
                         mention_sentence : str,
                         free_types: str,
                         coherence_mentions: str,
                         wiki_id: str
                         )-> Instance:

        sentence_tokenized = self.sentence_tokenizer.tokenize(mention_sentence)
        # sentence_field = TextField(sentence_tokenized, self.sentence_indexers)
        tokenized_left = [Token(START_SYMBOL)] + sentence_tokenized[1:st_idx+1] + [Token(END_SYMBOL)]
        tokenized_right = [Token(START_SYMBOL)] + sentence_tokenized[-2:et_idx+1:-1] + [Token(END_SYMBOL)]
        sentence_left_field = TextField(tokenized_left, self.sentence_indexers)
        sentence_right_field = TextField(tokenized_right, self.sentence_indexers)

        mention_surface_normalized = EnityLinknigDatasetReader.getLnrm(mention_surface)
        mention_normalized_field = MetadataField(mention_surface_normalized)

        types_field = MultiLabelField(labels=free_types.split(" "), label_namespace="type_labels", skip_indexing=False)
        coherence_tokenized = self.coherence_tokenizer.tokenize(coherence_mentions)
        coherences_field = TextField(coherence_tokenized, self.coherence_indexers)

        if wiki_id == "@@<unk_wid>@@":
            # because it is the test case and it packed by predictor class (demo) @@UNKNOWN@@ could be passed?
            if mention_surface_normalized in EnityLinknigDatasetReader.crosswiki:
                candidates = EnityLinknigDatasetReader.crosswiki[mention_surface_normalized][0]
                candidiate_probs = EnityLinknigDatasetReader.crosswiki[mention_surface_normalized][1]
                candidate_prob_field = ArrayField(np.array(candidiate_probs))
                candidates_tokenized = [Token(t) for t in candidates]
                candidates_field = TextField(candidates_tokenized, self.wid_indexers)
                # candidate_labels = [LabelField(label=t, label_namespace="wids") for t in candidates]
                # candidates_field = ListField(candidate_labels)

            else:
                candidate_prob_field = ArrayField( np.array([0.0]))
                # candidates_field = ListField([ LabelField(label="@@UNKNOWN@@", namespace="wids")])
                candidates_tokenized = [Token("@@UNKNOWN@@")]
                candidates_field = TextField(candidates_tokenized, self.wid_indexers)

            target_field = LabelField("label").empty_field()        # target_field.label is -1
        else:
            candidates = [wiki_id]
            if mention_surface_normalized in EnityLinknigDatasetReader.crosswiki:
                prior_wiki_ids = EnityLinknigDatasetReader.crosswiki[mention_surface_normalized][0] # crosswiki ( [candidates], [probs] )
                prior_wiki_probs = EnityLinknigDatasetReader.crosswiki[mention_surface_normalized][1]
                if wiki_id in prior_wiki_ids:
                    wiki_id_ind = prior_wiki_ids.index(wiki_id)
                    candidates = [wiki_id] + prior_wiki_ids[:wiki_id_ind] + prior_wiki_ids[wiki_id_ind+1:]
                    candidiate_probs = [prior_wiki_probs[wiki_id_ind]] + prior_wiki_probs[:wiki_id_ind] + prior_wiki_probs[wiki_id_ind+1:]
                else:
                    candidates = [wiki_id] + prior_wiki_ids
                    candidiate_probs = [0.0] + prior_wiki_probs

            if len(candidates) == 1:
                candidates += ["@@UNKNOWN@@"]
                candidiate_probs = [1.0, 0.0]

            candidate_prob_field = ArrayField(np.array(candidiate_probs))
            candidates_tokenized = [Token(t) for t in candidates]
            candidates_field = TextField(candidates_tokenized, self.wid_indexers)
            # candidate_labels = [LabelField(label=t, label_namespace="wids") for t in candidates]
            # candidates_field = ListField(candidate_labels)

            target_field = LabelField(label=0, label_namespace="label", skip_indexing=True)

        fields = {
            #"sentence": sentence_field,
            "sentence_left": sentence_left_field,
            "sentence_right": sentence_right_field,
            "mention_normalized": mention_normalized_field,
            "types": types_field,
            "coherences": coherences_field,
            "candidates": candidates_field,
            "candidate_priors": candidate_prob_field,
            # "candidate_id_meta": candidate_id_meta,     # possible to get id from indexer? should be...
            "targets": target_field,
        }

        return Instance(fields)     # no indexing no vocabulary
