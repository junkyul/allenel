from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('el-classifier')
class EnityLinknigPredictor(Predictor):
    # todo demo and evaluation
    pass
    # @overrides
    # def predict_json(self, json_dict: JsonDict) -> JsonDict:
    #     wid_label = json_dict["wid_label"]
    #     wiki_title_meta = json_dict["wiki_title_meta"]
    #     mention_surface_meta = json_dict["mention_surface_meta"]
    #     mention_sent_lt = json_dict["mention_sent_lt"]
    #     mention_sent_rt = json_dict["mention_sent_rt"]
    #     type_labels = json_dict["type_labels"]
    #     coherence_labels = json_dict["coherence_labels"]
    #     mention_sent_lt_len = json_dict["mention_sent_lt_len"]
    #     mention_sent_rt_len = json_dict["mention_sent_rt_len"]
    #     cross_wiki_candidates = json_dict["cross_wiki_candidates"]
    #     cross_wiki_priors = json_dict["cross_wiki_priors"]


