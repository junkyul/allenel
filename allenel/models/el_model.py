from typing import Dict, Optional, List
from overrides import overrides

from allennlp.models.model import Model
from allennlp.data import Vocabulary

from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward, Embedding, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask

import torch
# from torch.nn import Dropout, Embedding, CrossEntropyLoss, ReLU

import logging
logger = logging.getLogger(__name__)

@Model.register("el_model")
class EnityLinknigModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,  # Glove???
                 entity_embedder: TextFieldEmbedder,
                 coherence_embedder: TextFieldEmbedder,
                 word_embedding_dropout: float = 0.4,
                 coherence_embedding_dropout: float = 0.6,
                 left_seq2vec: Seq2VecEncoder = None,  # PytorchSeq2VecWrapper
                 right_seq2vec: Seq2VecEncoder = None,  # word drop out should be applied before embedding?
                 ff_seq2vecs : FeedForward = None,
                 ff_context: FeedForward = None,  # last feed forward layer [vm, vlocal]
                 initializer: InitializerApplicator = InitializerApplicator(),  # todo how to init normal 0, 0.01
                 regularizer: Optional[RegularizerApplicator] = None,  # todo how to use this?
                 )-> None:

        super(EnityLinknigModel, self).__init__(vocab, regularizer)
        self.vocab = vocab
        self.num_cohrences = self.vocab.get_vocab_size("cohereces")
        self.num_entities = self.vocab.get_vocab_size("wids")
        self.num_types = self.vocab.get_vocab_size("types")

        # local context
        self.sentence_embedder = sentence_embedder      # non-trainable need two embedders?
        self.word_embedding_dropout_left = torch.nn.Dropout(word_embedding_dropout)
        self.word_embedding_dropout_right = torch.nn.Dropout(word_embedding_dropout)
        self.left_seq2vec = left_seq2vec
        self.right_seq2vec = right_seq2vec
        self.ff_seq2vecs = ff_seq2vecs

        # coherence
        # self.coherence_embedder = coherence_embedder #Embedding(num_embeddings=1561683, embedding_dim=100, sparse=False)
        # self.coherence_embedder = Embedding(num_embeddings=coherence_embedding_opt['num_embeddings'],
        #                                              embedding_dim=coherence_embedding_opt['embedding_dim'],
        #                                              padding_idx=coherence_embedding_opt['padding_index'],
                                                     # sparse=coherence_embedding_opt['sparse'])
        self.coherence_embedder = coherence_embedder
        self.coherence_dropout = torch.nn.Dropout(coherence_embedding_dropout)
        self.coherence_embedder_relu = torch.nn.ReLU(inplace=False)

        # context
        self.ff_context = ff_context

        # entity
        # self.entity_embedder = entity_embedder #Embedding(num_embeddings=614129, embedding_dim=200, sparse=False)
        # self.entity_embedder = Embedding(num_embeddings=entity_embedding_opt['num_embeddings'],
        #                                           embedding_dim=entity_embedding_opt['embedding_dim'],
        #                                           padding_idx=entity_embedding_opt['padding_index'],
        #                                           sparse=entity_embedding_opt['sparse'])
        self.entity_embedder = entity_embedder

        # loss
        self.loss_context = torch.nn.CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            # "accuracy3": CategoricalAccuracy(top_k=2)
        }
        initializer(self)

    @overrides
    def forward(self,
                wid: List[str],
                title: List[str],
                types: torch.LongTensor,
                sentence: Dict[str, torch.LongTensor],              # shape B x seq_len x 300 (word2vec)
                sentence_left: Dict[str, torch.LongTensor],         # shape B x seq_len x 300 (word2vec)
                sentence_right: Dict[str, torch.LongTensor],
                mention: List[str],
                mention_normalized: List[str],
                coherences: Dict[str, torch.LongTensor],
                candidates: Dict[str, torch.LongTensor],
                targets: torch.LongTensor,
                )-> Dict[str, torch.Tensor]:
        # local context
        mask_left = get_text_field_mask(sentence_left)
        sentence_left_embedded = self.sentence_embedder(sentence_left)
        # logger.debug("sentence_left_embedded=Bx100? :{}".format(sentence_left_embedded.size()))
        sentence_left_dropped = self.word_embedding_dropout_left(sentence_left_embedded)
        # logger.debug("sentence_left_dropped=Bx100? :{}".format(sentence_left_dropped.size()))
        sentence_left_encoded = self.left_seq2vec(sentence_left_dropped, mask_left)
        # logger.debug("sentence_left_encoded=Bx100? :{}".format(sentence_left_encoded.size()))

        mask_right = get_text_field_mask(sentence_right)
        sentence_right_embedded = self.sentence_embedder(sentence_right)
        # logger.debug("sentence_right_embedded=Bx100? :{}".format(sentence_right_embedded.size()))
        sentence_right_dropped = self.word_embedding_dropout_right(sentence_right_embedded)
        # logger.debug("sentence_right_dropped=Bx100? :{}".format(sentence_right_dropped.size()))
        sentence_right_encoded = self.right_seq2vec(sentence_right_dropped, mask_right)
        # logger.debug("sentence_right_encoded=Bx100? :{}".format(sentence_right_encoded.size()))

        sentence_encoded = torch.cat((sentence_left_encoded, sentence_right_encoded), dim=1)
        # logger.debug("sentence_encoded=Bx100? :{}".format(sentence_encoded.size()))
        local_context = self.ff_seq2vecs(sentence_encoded)
        # logger.debug("local_context=Bx100? :{}".format(local_context.size()))

        # coherence
        coherence_embedded = self.coherence_embedder(coherences)
        # logger.debug("coherence_embedded=Bx100? :{}".format(coherence_embedded.size()))     # B x C x 100
        coherence_embedded = torch.sum(coherence_embedded, dim=1).view(coherence_embedded.size()[0], -1)        # B x
        # logger.debug("coherence_embedded=Bx100? :{}".format(coherence_embedded.size()))
        coherence_dropped = self.coherence_dropout(coherence_embedded)
        # logger.debug("coherence_dropped=Bx100? :{}".format(coherence_dropped.size()))
        coherence_embedded_relu = self.coherence_embedder_relu(coherence_dropped)
        # logger.debug("coherence_embedded_relu=Bx100? :{}".format(coherence_embedded_relu.size()))

        # coherence_embedded = torch.matmul(coherences, )

        # context
        v_local = torch.cat( (local_context, coherence_embedded_relu), dim=1)
        # logger.debug("v_local=Bx200? :{}".format(v_local.size()))
        v_local = self.ff_context(v_local)
        # logger.debug("v_local=Bx200? :{}".format(v_local.size()))
        v_local = v_local.view(v_local.size()[0], 1, -1)
        # logger.debug("v_local=B x 1 x 200? :{}".format(v_local.size()))

        # entity
        candidates_embedded = self.entity_embedder(candidates)    # first element is true label
        # logger.debug("candidates_embedded=B x max_labels_padded x 200? :{}".format(candidates_embedded.size()))


        scores = torch.matmul(candidates_embedded, v_local.view(v_local.size()[0], -1, 1) )
        # logger.debug("scores=B x C? :{}".format(scores.size()))
        scores = torch.squeeze(scores)
        # logger.debug("scores=B x C? :{}".format(scores.size()))
        # logger.debug("targets=B x C? :{}".format(targets.size()))
        loss = self.loss_context(scores, targets)

        with torch.no_grad():
            # probs = torch.nn.functional.softmax(scores, dim=-1)
            for metric in self.metrics.values():
                metric(scores, targets)

        output_dict = {'loss': loss, "scores": scores}       # use "scores" to compute softmax probability
        return output_dict

    # @overrides
    # def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    #     todo decode numeric 'wid' to wiki title for demo or create a link https://en.wikipedia.org/?curid=<wid>
        # return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}