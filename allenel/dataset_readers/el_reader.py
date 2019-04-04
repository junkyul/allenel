from typing import Dict
from array import ArrayType
import json

from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, LabelField, ListField, ArrayField, TextField
from allennlp.data import Token
from allennlp.data.instance import Instance
from .el_multiproc_reader import EnityLinknigDatasetMultiReader
import random
import numpy as np

import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("el_reader")
class EnityLinknigDatasetReader(DatasetReader):
    def __init__(self,
                 resource_path: str = "",
                 word_drop_sentence : float = 0.4,
                 word_drop_coherence: float = 0.6,
                 ) -> None:
        super().__init__(lazy=True)
        self.word_drop_sentence= word_drop_sentence
        self.word_drop_coherence = word_drop_coherence
        self.pickle_loaded = False


        if resource_path:
            import pickle, os
            logger.info("reading knw_wid_vocab.pkl")
            kwn_wid_vocab = pickle.load(open(os.path.join(resource_path, "vocab/knwn_wid_vocab.pkl"), "rb"))
            logger.info("reading cohstringG9_vocab.pkl")
            conherence_vocab = pickle.load(open(os.path.join(resource_path, "vocab/cohstringG9_vocab.pkl"), "rb"))
            logger.info("reading label_vocab.pkl")
            type_vocab = pickle.load(open(os.path.join(resource_path, "vocab/label_vocab.pkl"), "rb"))
            logger.info("reading glove_word_vocab.pkl")
            glove_vocab = pickle.load(open(os.path.join(resource_path, "vocab/glove_word_vocab.pkl"), "rb"))
            self.wid_index = kwn_wid_vocab[0]
            self.coherence_index = conherence_vocab[0]
            self.type_index = type_vocab[0]
            self.glove_index = glove_vocab[0]
            logger.info("reading glove.pkl")
            self.glove_embedding = pickle.load(open(os.path.join(resource_path, "glove.pkl"), "rb"))
            logger.info("reading crosswikis.pruned.pkl")
            self.crosswiki_priors = pickle.load(open(os.path.join(resource_path, "crosswikis.pruned.pkl"), "rb"))
            logger.info("reading wid2Wikititle.pkl")
            self.title_dict = pickle.load(open(os.path.join(resource_path,"vocab/wid2Wikititle.pkl"), "rb"))
            self.pickle_loaded = True

    @overrides
    def _read(self, file_path):
        if not self.pickle_loaded:
            # access from MultiReader class (load once from MultiReader)
            self.wid_index = EnityLinknigDatasetMultiReader.wid_index
            self.coherence_index = EnityLinknigDatasetMultiReader.coherence_index
            self.type_index = EnityLinknigDatasetMultiReader.type_index
            self.crosswiki_priors = EnityLinknigDatasetMultiReader.crosswiki_priors
            self.glove_index = EnityLinknigDatasetMultiReader.glove_index
            self.title_dict = EnityLinknigDatasetMultiReader.title_dict
            self.glove_embedding = EnityLinknigDatasetMultiReader.glove_embedding


        with open(file_path) as data_file:
            for line in data_file:
                line = line.strip("\n")
                if not line: continue
                temp = line.split("\t")
                free_id, wiki_id, wiki_title = temp[0], temp[1], temp[2]
                st_idx, et_idx, mention_surface, mention_sentence= temp[3], temp[4], temp[5], temp[6]
                free_types = temp[7] if len(temp) > 7 else ""
                coherence_mentions = temp[8] if len(temp) > 8 else ""
                yield self.text_to_instance(wiki_id.strip(), wiki_title.strip(), int(st_idx.strip()), int(et_idx.strip()),
                                            mention_surface.strip(), mention_sentence.strip(),
                                            free_types.strip(), coherence_mentions.strip())
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

        cur_wid_ind = self.wid_index[wiki_id]
        wiki_id_field = LabelField(label=cur_wid_ind, skip_indexing=True,
                                   label_namespace='wid_labels')        # one label
        wiki_title_field = MetadataField(metadata=wiki_title)
        mention_surf_norm = EnityLinknigDatasetReader.getLnrm(mention_surface)
        mention_surface_field = MetadataField(metadata=mention_surf_norm)

        cross_wiki_candidates, cross_wiki_priors = [], []
        if mention_surf_norm in self.crosswiki_priors:
            cross_wiki_candidates, cross_wiki_priors = self.crosswiki_priors[mention_surf_norm]
            cross_wiki_candidates = [self.wid_index[t] for t in cross_wiki_candidates]
            if len(cross_wiki_candidates) != len(cross_wiki_priors):
                cross_wiki_priors = cross_wiki_priors[:len(cross_wiki_candidates)]

        else:
            cross_wiki_candidates = [cur_wid_ind]
            cross_wiki_priors = [0.0]

        if cur_wid_ind in cross_wiki_candidates:
            for t in range(len(cross_wiki_candidates)):
                if cross_wiki_candidates[t] == cur_wid_ind:
                    cross_wiki_candidates[0], cross_wiki_candidates[t] = cross_wiki_candidates[t], cross_wiki_candidates[0]
                    cross_wiki_priors[0], cross_wiki_priors[t] = cross_wiki_priors[t], cross_wiki_priors[0]
                    break
        else:   # not in candidate cur_wid
            cross_wiki_candidates = [cur_wid_ind] + cross_wiki_candidates
            cross_wiki_priors = [0.0] + cross_wiki_priors

        # append unk_wid with probability zero as candidate to avoid one class issue
        if len(cross_wiki_candidates) == 1:
            cross_wiki_candidates.append(0)
            cross_wiki_priors.append(0)

        cross_wiki_candidates_field = MetadataField(cross_wiki_candidates)
        cross_wiki_priors_field = MetadataField(cross_wiki_priors)

        if self.word_drop_sentence > 0.0:
            sentence_tokens = []
            for t in mention_sentence.split(" "):
                sentence_tokens.append(t if random.random() > self.word_drop_sentence else "unk")
        else:
            sentence_tokens = mention_sentence.split(" ")
        sentence_tokens = ["<s>"] + sentence_tokens + ["<eos>"]

        sentence_array = []
        for t in sentence_tokens:
            if t in self.glove_embedding:
                sentence_array.append(self.glove_embedding[t])
            elif t == "<eos>":
                sentence_array.append(self.glove_embedding["eos"])
            else:
                sentence_array.append(self.glove_embedding["unk"])
        sentence_l_field = ArrayField(np.array(sentence_array[:st_idx+1]), padding_value=0.0)
        sentence_r_field = ArrayField(np.array(sentence_array[-1:et_idx+1:-1]), padding_value=0.0)

        sentence_l_len_field = MetadataField(metadata=len(sentence_array[:st_idx+1]))
        sentence_r_len_field = MetadataField(metadata=len(sentence_array[-1:et_idx+1:-1]))

        # sentence_l = ["<s>"] + sentence_tokens[:st_idx]
        # sentence_l_tokens = []
        # for t in sentence_l:
        #     if t in self.glove_index:
        #         sentence_l_tokens.append(Token(text=t, text_id=self.glove_index[t]))
        #     else:
        #         sentence_l_tokens.append(Token(text="unk", text_id=self.glove_index["unk"]))
        # sentence_l_field = TextField(sentence_l_tokens, token_indexers={})
        #
        # sentence_r = ["<eos>"] + sentence_tokens[-1:et_idx:-1]
        # sentence_r_tokens = []
        # for t in sentence_r:
        #     if t in self.glove_index:
        #         sentence_r_tokens.append(Token(text=t, text_id=self.glove_index[t]))
        #     else:
        #         sentence_r_tokens.append(Token(text="unk", text_id=self.glove_index["unk"]))
        # sentence_r_field = TextField(sentence_r_tokens, token_indexers={})

        # type_labels= [LabelField(self.type_index[t], skip_indexing=True,
        #                          label_namespace='type_labels') for t in free_types.split(" ")]
        # types_field = ListField(type_labels)
        type_array = np.zeros(113, dtype=int)
        type_array[ [self.type_index[t] for t in free_types.split(" ")] ] = 1
        types_field = ArrayField(type_array, padding_value=-1)

        coherence_labels = set()
        for t in coherence_mentions.split(" "):
            if t in self.coherence_index:
                if random.random() > self.word_drop_coherence:
                    temp  = t
                else:
                    temp = "unk"
                coherence_labels.add(self.coherence_index[temp])
                # coherence_labels.append( LabelField(self.coherence_index[temp], skip_indexing=True,
                #                                     label_namespace='coherence_labels') )
            else:
                # coherence_labels.append(
                #     LabelField(self.coherence_index["unk"], skip_indexing=True,
                #                label_namespace='coherence_labels'))
                coherence_labels.add(self.coherence_index["unk"])
        coherences_field = MetadataField(sorted(list(coherence_labels)))
        # coherences_field = ListField(coherence_labels)
        # coherences_field = ArrayField(np.array(coherence_labels, dtype=np.int32), padding_value=-1)

        fields = {
            'wid_label': wiki_id_field,            # 1 label out of... 620k
            'wiki_title_meta': wiki_title_field,
            'mention_surface_meta': mention_surface_field,
            'mention_sent_lt': sentence_l_field,    # list of array
            'mention_sent_rt': sentence_r_field,
            'type_labels': types_field,             # list[labels] each label 113
            'coherence_labels': coherences_field,   # list[labels] each label from 1.5E+6,
            'mention_sent_lt_len': sentence_l_len_field,
            'mention_sent_rt_len': sentence_r_len_field,
            'cross_wiki_candidates': cross_wiki_candidates_field,
            'cross_wiki_priors': cross_wiki_priors_field
        }
        return Instance(fields)     # no indexing no vocabulary


        # mention_sentence_tokenized = self._tokenizer.tokenize(mention_sentence)
        # mention_sentence_left = TextField(mention_sentence_tokenized[:st_idx + 1], self._token_indexers)
        # mention_sentence_right = TextField(mention_sentence_tokenized[-1:et_idx + 1:-1], self._token_indexers)
        # [LabelField() free_types.split(" ")



        # convert labels to int (skip_indexing) coherence masking (dropout)


        # class EnityLinknigDatasetMultiReader(MultiprocessDatasetReader):
        #     wid_index: Dict[str, int]
        #     coherence_index: Dict[str, int]
        #     type_index: Dict[str, int]
        #     crosswiki_priors: Dict[str, tuple]
        #     glove_index: Dict[str, int]
        #     title_dict: Dict[str, str]
        #     glove_embedding: Dict[str, ArrayType]

        # free_types_field = MultiLabelField(labels=free_types_tokenized,
        #                                    label_namespace="types", skip_indexing=True, num_labels=113)
        # coherence_field = MultiLabelField(labels=coherence_mentions_tokenized,
        #                                   label_namespace="coherences", skip_indexing=True, num_labels=1561683)




        # no tokenizer required for the current setup!
        # there's no simple way

        # what to init?

        # JustSpacesWordSplitter() only use space to split tokens when creating a field
        # self._tokenizer = tokenizer or WordTokenizer(word_splitter=JustSpacesWordSplitter())

        # Indexer does not actually give num ids until it is called and provided with a global Vocabulary
        # singleid_token_indexer = SingleIdTokenIndexer(namespace="tokens", start_tokens=["<s>"], end_tokens=["<eos>"])
        # self._token_indexers = token_indexers or {"tokens": singleid_token_indexer}
        # word level vs char level...
        # labels don't need token indexer..?
        # this indexer is for... word tokens, char tokens, etc


        # how to connect Vocabs with the pickle files?
        # when Vocab is used to transform tokens to num indices?
        # hot wo pass pickle objects to Vocabs?
        # where do I define Vocabs? when it actually calls... Vocabs pass it



    # """
    # Reads resource files
    # pickle files for
    #     GloVe Word Vocab: (Dict[word, index], List[word]) used for tokens in a sentence
    #     Label Vocab: (Dict[type_label, index], List[type_label[)    used for types
    #     Known WID Vocab:   (Dict[wiki_num_id, index], List[wiki_num_id])
    #     WID to title:   Dict[WID, string title]     for printing out results
    #     Coherence strings:  (Dict[coh_string, index], List[coh_string[) for coherence information
    #     Glove Vector:   Dict[word, 200 dim array]
    #
    # Training data (o) for actual data
    #     each line separated by tab
    #     line[0] Freebase ID
    #     o line[1] WID
    #     line[2] WID2title
    #     line[3] Mention surface start index             add "<s>"
    #     line[4] Mention surface end index (inclusive)    add "<eos>"
    #     line[5] Mention surface as a whole string
    #     o line[6] Tokenized sentence separated by space
    #     o line[7] Tokenized type labels separated by space
    #     o line[8] Tokenized coherence string labels separated by space   if empty, "<unk_word>"
    #
    #     define name space for the data indexer
    #     wids (labels), tokens, types (labels), coherences (labels)
    #
    # the data set is large for desktop machines (~ 30 GB training examples, ~ 30 GB resource files)
    #     loading 30 GB resource file looks inevitable
    #     (Q) if we don't use the pre-trained word embeddings, should we load 30 GB at once to AllenNLP?
    #         - 30 GB training files are divided into 41 files
    #         - does AllenNLP builds a Vocab for the whole example after scanning things at once?
    #         - is it required to scan whole example before training?
    #         - is it possible to build a vocab and do training at the same time?
    #
    #     (Q) lazy true/false; if true _read() returns iter else returns a list
    #     (A) @DatasetReader.register('multiprocess')
    #         class MultiprocessDatasetReader(DatasetReader)
    #
    #         (Q) Does MultiprocessDatasetReader handles multi-epochs smoothly?
    #             41 files read one or two and then finish 1 epoch then
    #             the DataIterator will call Reader to start from the first file?
    #         (?) Should be... or why it's there
    #
    # """
