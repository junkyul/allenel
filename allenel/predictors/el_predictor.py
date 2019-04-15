from typing import List
import json

from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from ccg_nlpy import remote_pipeline

from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

import logging
logger = logging.getLogger(__name__)


@Predictor.register('el_linker')
class EnityLinknigPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)
        self.pipeline = remote_pipeline.RemotePipeline(server_api='http://macniece.seas.upenn.edu:4001')

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instances = self._json_to_instances(inputs)

        return_dicts: List[JsonDict] = [ {"entities":[], "scores":[]} for _ in instances]
        for instance_ind, each_instance in enumerate(instances):
            output_dict = self.predict_instance(each_instance)      # return from forward on the instance
            p_texts = output_dict['scores']
            candidate_ids = [self._model.vocab.get_token_from_index(index=ind, namespace="wids") for ind in output_dict['candidates']]

            mention_normalized = output_dict['mention_normalized']
            if mention_normalized in self._dataset_reader.crosswiki:
                prior_ids = self._dataset_reader.crosswiki[mention_normalized][0]
                prior_probs = self._dataset_reader.crosswiki[mention_normalized][1]
                prior_dict = {prior_id: prior_probs[i] if i < len(prior_probs) else 0.0 for i, prior_id in enumerate(prior_ids)}
                p_post = []
                for i, cand_id in enumerate(candidate_ids):
                    tmp_prior = prior_dict[cand_id] if cand_id in prior_dict else 0.0
                    p_post.append( p_texts[i] + tmp_prior - (p_texts[i] * tmp_prior) )
            else:
                p_post = p_texts

            scores, entities = ( list(t) for t in zip (*sorted(zip(p_post, candidate_ids))) )
            return_dicts[instance_ind]["entities"] = entities[:3]
            return_dicts[instance_ind]["scores"] = scores[:3]
            return_dicts[instance_ind]["input_data"] = self.input_data[instance_ind]
        return sanitize(return_dicts)

    def _json_to_instances(self, json_dict: JsonDict) -> List[Instance]:
        """ returns a list of instances """
        sentence_raw = json_dict['context'].strip()
        sentence_tokenized = self._tokenizer.split_words(sentence_raw)
        self._process_test_doc(" ".join([t.text for t in sentence_tokenized]))
        test_cases = self.convertSent2NerToMentionLines()        # returns structured format as training data
        instances = []
        self.input_data = []
        for each_case in test_cases:
            tmp = self._dataset_reader.process_line(each_case)
            self.input_data.append({"start": tmp[0], "end": tmp[1], "mention": tmp[2], "sentence": tmp[3]})
            instances.append( self._dataset_reader.text_to_instance(*tmp) )
        return instances

    def _process_test_doc(self, sentence_raw):
        """ taken from processTestDoc from neural-el project by Nitish Gupta """
        self.doctext = sentence_raw
        self.ccgdoc = self.pipeline.doc(self.doctext)
        self.doc_tokens = self.ccgdoc.get_tokens
        self.sent_end_token_indices = self.ccgdoc.get_sentence_end_token_indices
        self.sentences_tokenized = []
        for i in range(0, len(self.sent_end_token_indices)):
            start = self.sent_end_token_indices[i-1] if i != 0 else 0
            end = self.sent_end_token_indices[i]
            sent_tokens = self.doc_tokens[start:end]
            self.sentences_tokenized.append(sent_tokens)

        # List of ner dicts from ccg pipeline
        self.ner_cons_list = []
        try:
            self.ner_cons_list = self.ccgdoc.get_ner_conll.cons_list
        except:
            print("NO NAMED ENTITIES IN THE DOC. EXITING")

        self.sentidx2ners = {}
        for ner in self.ner_cons_list:
            found = False
            # idx = sentIdx, j = sentEndTokenIdx
            for idx, j in enumerate(self.sent_end_token_indices):
                sent_start_token = self.sent_end_token_indices[idx-1] \
                    if idx != 0 else 0
                # ner['end'] is the idx of the token after ner
                if ner['end'] < j:
                    if idx not in self.sentidx2ners:
                        self.sentidx2ners[idx] = []
                    ner['start'] = ner['start'] - sent_start_token
                    ner['end'] = ner['end'] - sent_start_token - 1
                    self.sentidx2ners[idx].append(
                        (self.sentences_tokenized[idx], ner))
                    found = True
                if found:
                    break

    def convertSent2NerToMentionLines(self):
        '''Convert NERs from document to list of mention strings'''
        mentions = []
        # Make Document Context String for whole document
        cohStr = ""
        for sent_idx, s_nerDicts in self.sentidx2ners.items():
            for s, ner in s_nerDicts:
                cohStr += ner['tokens'].replace(' ', '_') + ' '
        cohStr = cohStr.strip()

        for idx in range(0, len(self.sentences_tokenized)):
            if idx in self.sentidx2ners:
                sentence = ' '.join(self.sentences_tokenized[idx])
                s_nerDicts = self.sentidx2ners[idx]
                for s, ner in s_nerDicts:
                    mention = "%s\t%s\t%s" % ("unk_mid", "@@UNKNOWN@@", "unkWT")
                    mention = mention + str('\t') + str(ner['start'])
                    mention = mention + '\t' + str(ner['end'])
                    mention = mention + '\t' + str(ner['tokens'])
                    mention = mention + '\t' + sentence
                    mention = mention + '\t' + "@@UNKNOWN@@"
                    cur_coh = set(cohStr.split())
                    cur_coh.remove( ner['tokens'].replace(' ', '_'))
                    mention = mention + '\t' + " ".join(cur_coh)
                    mentions.append(mention)
        return mentions


