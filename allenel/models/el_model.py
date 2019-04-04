from typing import Dict, Optional, List

from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder, FeedForward, Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy

import numpy as np
from overrides import overrides
import torch
from torch.nn import Embedding, CrossEntropyLoss, ReLU

import logging
logger = logging.getLogger(__name__)

@Model.register("el_model")
class EnityLinknigModel(Model):
    def __init__(self,
                 left_seq2vec: Seq2VecEncoder = None,       # PytorchSeq2VecWrapper
                 right_seq2vec: Seq2VecEncoder = None,      # word drop out should be applied before embedding?
                 ff_seq2vecs : FeedForward = None,
                 ff_context: FeedForward = None,            # last feed forward layer [vm, vlocal]
                 ff_type: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),      # todo how to init normal 0, 0.01
                 regularizer: Optional[RegularizerApplicator] = None,               # todo how to use this?
                 use_coherence: bool = False,
                 use_type: bool = False,
                 device: str = "cpu",
                 )-> None:
        super(EnityLinknigModel, self).__init__({}, regularizer)
        self.use_coherence = use_coherence
        self.use_type = use_type
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # context
        self.left_seq2vec = left_seq2vec
        self.right_seq2vec = right_seq2vec
        self.ff_seq2vecs = ff_seq2vecs
        self.ff_context = ff_context
        if self.use_coherence:
            assert ff_context is not None
            self.coherence_embedder = Embedding(num_embeddings=1561683, embedding_dim=100, sparse=False)
            self.coherence_embedder_relu = ReLU(inplace=False)

        # entity
        self.entity_embedder = Embedding(num_embeddings=614129, embedding_dim=200, sparse=False)

        # type
        self.ff_type = ff_type
        if self.use_type:
            assert ff_type is not None

        # losses
        self.loss_context = CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=2)
        }

        initializer(self)
        # https://github.com/allenai/allennlp/blob/master/training_config/srl_elmo_5.5B.jsonnet
        # https://github.com/allenai/allennlp/blob/master/allennlp/models/semantic_role_labeler.py

        # todo resume training
        # https://github.com/allenai/allennlp/blob/master/training_config/bidirectional_language_model.jsonnet#L53
        # how to pack a model from.. .th file?

    def make_mask(self, batch_size: int, max_len: int, seq_len: List[int])-> torch.Tensor:
        mask = np.zeros( (batch_size, max_len))
        for i in range(batch_size):
            mask[i, :seq_len[i]] = np.ones(seq_len[i])
        return torch.from_numpy(mask).to(self.device)

    @overrides
    def forward(self,
                wid_label: torch.LongTensor,        # shape B   string number -> int number
                wiki_title_meta: List[str],         # len B list
                mention_surface_meta: List[str],    # len B list
                mention_sent_lt: torch.FloatTensor, # shape B x seq_len x 300 (word2vec)
                mention_sent_rt: torch.FloatTensor, # shape B x seq_len x 300 (word2vec)
                type_labels: torch.LongTensor,      # shap B x seq_len --> dense one hot vector from reader [no padding]
                coherence_labels: List[List[int]],  #
                mention_sent_lt_len: List[int],
                mention_sent_rt_len: List[int],
                cross_wiki_candidates: List[List[int]],
                cross_wiki_priors: List[List[float]]
                ):
        batch_size = wid_label.shape[0]
        mask_left = self.make_mask(batch_size, mention_sent_lt.shape[1], mention_sent_lt_len)
        sent_lt_encoded = self.left_seq2vec(mention_sent_lt, mask=mask_left)         # build mask manually for ArrayField

        mask_right = self.make_mask(batch_size, mention_sent_rt.shape[1],mention_sent_rt_len)
        sent_rt_encoded = self.right_seq2vec(mention_sent_rt, mask=mask_right)

        sent = torch.cat( (sent_lt_encoded, sent_rt_encoded), dim=1).to(self.device)         # dim0 is batch no adding batch

        v_local = self.ff_seq2vecs(sent)               # B x 200 --> B x 100

        if self.use_coherence:
            v_coh_batch = torch.zeros( (batch_size, 100)).to(self.device)
            for i in range(batch_size):
                coherence_ind = torch.LongTensor(coherence_labels[i]).to(self.device)
                v_coh1 = self.coherence_embedder(coherence_ind)
                v_coh2 = torch.sum(v_coh1, dim=0).view(1, -1).to(self.device)
                v_coh_batch[i,:] = v_coh2
            v_coh_batch = self.coherence_embedder_relu(v_coh_batch)       # todo check

            v_local = torch.cat( (v_local, v_coh_batch), dim=1 )

            v_local = self.ff_context(v_local)

        if self.use_type:
            v_type = self.ff_type(type_labels)

        loss = 0.0
        v_e_list = []
        for i in range(batch_size):
            entity_id = torch.LongTensor(cross_wiki_candidates[i]).to(self.device)       # 1 x C (C > 1)
            # this C, num classes changes per example pad zero/unk_wid?
            target = torch.LongTensor([0]).to(self.device)
            v_e = self.entity_embedder(entity_id)      # C x 200
            v_local_cur = v_local[i].view(-1, 1)       # 200 x 1
            score = torch.matmul(v_e, v_local_cur).view(1, -1)     # 1 x C
            temp = self.loss_context(score, target)
            loss += temp
            v_e_list.append(v_e)

        output_dict = {'loss': loss} #, 'v_local': v_local, 'wid_label': wid_label }
                # 'candidates': cross_wiki_candidates, 'priors': cross_wiki_priors,
                # , 'v_e_list': v_e_list }

        # compute accuray metrics
        #with torch.no_grad():
        #    max_candidates = max( (len(t) for t in cross_wiki_candidates) )
        #    predictions_per_batch = []
        #    true_labels = wid_label
        #    for i in range(batch_size):   # i is for batch
        #        # list of prior probabilities len > 1    list 1 x C
        #        cur_prior = torch.FloatTensor(cross_wiki_priors[i]).view(1, -1).to(self.device)
        #        cur_ve = v_e_list[i].view(-1, 200)             # C x 200       tensor
        #        cur_vm = v_local[i].view(200, 1)            # 200 x 1    tensor

        #        prob_text = torch.exp(torch.matmul(cur_ve, cur_vm)).view(-1, 1)
        #        prob_text = prob_text / torch.sum(prob_text).to(self.device)     # C x 1

        #        temp = torch.zeros(max_candidates).to(self.device)
        #        prob_text = torch.squeeze(prob_text)
        #        cur_prior = torch.squeeze(cur_prior)
        #        temp[:len(cur_prior)] = cur_prior + prob_text - cur_prior * prob_text       # when this value can be nan?

        #        predictions_per_batch.append(temp)

        #    predictions = torch.stack(predictions_per_batch)
        #    for metric in self.metrics.values():
        #        metric(predictions=predictions,
        #               gold_labels=torch.zeros(len(true_labels)))  # true label is located at zero
        return output_dict

    #@overrides
    #def decode(self, output_dict: Dict[str, torch.Tensor])-> Dict[str, torch.Tensor]:
    #    # todo decode numeric 'wid' to wiki title for demo or create a link https://en.wikipedia.org/?curid=<wid>
    #    return output_dict


    #@overrides
    #def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #    return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

        # how to compute the loss?

        # self.type_ff = type_ff
        # # this is GloVe provided by external method... match format
        # self.context_embedder = context_embedder        # pre-define embedder with params...
        #
        # # dropout applies before embedding
        # self.context_embedder_dropout = Dropout(p=word_dropout_mention)     # word2 vec instantiate from config
        # # replace several words by unk
        #
        # self.entity_embedder = Embedding(num_embeddings=614129,
        #                                  embedding_dim=200,
        #                                  trainable=True,
        #                                  sparse=False)     # predefined vocab indexer?
        #
        # self.coherece_embedder = Embedding(num_embeddings=1561683,
        #                                    embedding_dim=100,
        #                                    trainable=True,
        #                                    sparse=True,
        #                                    vocab_namespace="coherences")
        # self.coherence_encoder_dropout = Dropout(p=word_dropout_coherence)
        # # replace several mentions by unk
        #
        # self.type_embedder = Embedding(num_embeddings=113,
        #                                embedding_dim=200,
        #                                trainable=True,
        #                                sparse=False,
        #                                vocab_namespace="types")
        #
        # # drop out for embedding??? 1 layer LSTM no dropout
        # # https://github.com/allenai/allennlp/blob/master/allennlp/models/coreference_resolution/coref.py#L70
        # # if mention_dropout > 0.0:
        # #     self.mention_dropout = Dropout(p=mention_dropout)
        # # else:
        # #     self.mention_dropout = lambda  x: x
        #
        #
        # # seq2vec
        # # self.mention_encoder = mention_encoder
        # self.left_words_seq2vec = left_words_seq2vec
        # self.right_words_seq2vec = right_words_seq2vec
        # self.left_right_vecs_ff = left_right_vecs_ff
        #
        # # sparse FF
        # self.coherence_encoder = coherence_ff
        #
        #
        # # dense FF
        # self.type_encoder = type_ff
        #
        # # how to make a judgement?

        #
        # # define a class later; wrap CrossEntropy
        # # there was an example of this kind
        # self.loss = torch.nn.CrossEntropyLoss()

        # convert passed data from str num to int
        # pick up sub matrix of entity embedding by.. using wid_labels + mentions_surface_meta [cross wiki max 30 candidates]

        # entity_label_list = [wid_label]
        # entity_label_embeddings = self.entity_embedder(entity_label_list)       # index zero is <unk_wid> padding zero ok?? at least prediction time dont do this

        # add v_e embedding (dense) and then cross entropy loss

        # logits : torch.FloatTensor
        #     A tensor of shape ``(batch_size, num_labels)`` representing
        #     unnormalized log probabilities of the label.
        # probs : torch.FloatTensor
        #     A tensor of shape ``(batch_size, num_labels)`` representing
        #     probabilities of the label.
        # loss : torch.FloatTensor, optional
        #     A scalar loss to be optimised.

        # https: // allennlp.org / tutorials
        # lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

        # "context_layer": {
        #     "type": "lstm",
        #     "bidirectional": true,
        #     "input_size": 20,
        #     "hidden_size": 10,
        #     "num_layers": 1
        # },
        # "mention_feedforward": {
        #     "input_dim": 65,
        #     "num_layers": 1,
        #     "hidden_dims": 10,
        #     "activations": "relu"
        # },


        # return from forward is the final outcome of NN
        # in this case, it is... v_m        because predictor will use v_m
        # semi-supervised learning
        # learn entity embedding -> lower dim representation of entities
        # v_e output for each entity
        # v_m output from mention sentence
        # v_t output from type

        # as a result of training, by backprop loss function
        # learn v_e matrix and all other params
        # it looks the optimization is loose...but it will settle down somewhere
        # at the stage of prediction, use v_e and v_m
        # we don't use... v_t for prediction....
        # conversly, do we predict type labels from v_m? input sentence?
        # generate description? ??? by lang model???


        # the params of forward... matches to the namespace of fields dictionary in Instance
        # this will apply to the Vocabulary

        # all inputs pass embedding

        # mention sentence
        #   seq of Tokens... [how to convert to ints? here or before by using Vocal]
        #   -> [GloVe embedding, predefined, word2 300 dim vec]
        #   -> 2 LSTMs seq2vec; abstract once embedded or 1 LSTM output 100 dim
        #   -> concatenate output 200 dim
        #   -> FFN
        #   -> ReLu     -> v_local 100 dim

        # coherences
        #   -> one hot of labels.. large dim..! MultiLabel OK??
        #   -> sparse embedding output 100 dim
        #   -> ReLu     -> v_global 100 dim

        # types (make this optional)
        #   -> one hot of labels... 113 dim dense MultiLabel OK??
        #   -> dense embedding output 200 dim
        #   -> sigmoid  -> v_type 200 dim


