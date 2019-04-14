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
                 sentence_embedder: TextFieldEmbedder,
                 entity_embedder: TextFieldEmbedder,
                 coherence_embedder: TextFieldEmbedder,
                 encoded_dims: int = 200,
                 word_embedding_dropout: float = 0.4,
                 coherence_embedding_dropout: float = 0.6,
                 left_seq2vec: Seq2VecEncoder = None,
                 right_seq2vec: Seq2VecEncoder = None,
                 ff_seq2vecs : FeedForward = None,
                 ff_context: FeedForward = None,
                 type_embedder: TokenEmbedder = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 model_modules = "C"        # by default C model C, CT, CE, CTE,
                 )-> None:

        super(EnityLinknigModel, self).__init__(vocab, regularizer)
        self.encoded_dims = encoded_dims
        # context sentence
        self.sentence_embedder = sentence_embedder
        self.word_embedding_dropout_left = torch.nn.Dropout(word_embedding_dropout)
        self.word_embedding_dropout_right = torch.nn.Dropout(word_embedding_dropout)
        self.left_seq2vec = left_seq2vec
        self.right_seq2vec = right_seq2vec
        self.ff_seq2vecs = ff_seq2vecs
        # context coherent mentions
        self.coherence_embedder = coherence_embedder
        self.coherence_dropout = torch.nn.Dropout(coherence_embedding_dropout)
        self.coherence_embedder_relu = torch.nn.ReLU(inplace=False)
        self.ff_context = ff_context

        # entity
        self.entity_embedder = entity_embedder

        # type labels
        self.type_embedder = type_embedder

        # loss
        self.loss_text = torch.nn.CrossEntropyLoss()
        # equivalent to tf.sigmoid_cross_entropy_with_logits
        self.loss_etype = torch.nn.BCEWithLogitsLoss() if "E" in model_modules and self.type_embedder else None
        self.loss_mtype = torch.nn.BCEWithLogitsLoss() if "T" in model_modules and self.type_embedder else None

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        initializer(self)

        if self.loss_etype and self.loss_mtype:
            logger.info("Initialized CTE entity linking model")
        elif self.loss_etype:
            logger.info("Initialized CE entity linking model")
        elif self.loss_mtype:
            logger.info("Initialized CT entity linking model")
        else:
            logger.info("Initialized C entity linking model")

    @overrides
    def forward(self,
                sentence_left: Dict[str, torch.LongTensor],         # [B, sent len, 300]
                sentence_right: Dict[str, torch.LongTensor],
                mention: List[str],
                mention_normalized: List[str],
                candidates: Dict[str, torch.LongTensor],
                types: Dict[str, torch.LongTensor] = None,
                coherences: Dict[str, torch.LongTensor] = None,
                targets: torch.LongTensor = None,
                )-> Dict[str, torch.Tensor]:
        # local context
        mask_left = get_text_field_mask(sentence_left)
        sentence_left = self.sentence_embedder(sentence_left)
        sentence_left = self.word_embedding_dropout_left(sentence_left)
        sentence_left = self.left_seq2vec(sentence_left, mask_left)

        mask_right = get_text_field_mask(sentence_right)
        sentence_right = self.sentence_embedder(sentence_right)
        sentence_right = self.word_embedding_dropout_right(sentence_right)
        sentence_right = self.right_seq2vec(sentence_right, mask_right)

        sentence_encoded = torch.cat((sentence_left, sentence_right), dim=1)
        local_context = self.ff_seq2vecs(sentence_encoded)      # [B, 100]

        # coherence
        coherences = self.coherence_embedder(coherences)
        coherences = torch.sum(coherences, dim=1).view(coherences.shape[0], -1)
        coherences = self.coherence_dropout(coherences)
        coherences = self.coherence_embedder_relu(coherences)   # [B, 100]

        # context
        v_local = torch.cat( (local_context, coherences), dim=1)
        v_local = self.ff_context(v_local)  #[B, 200]
        v_local = v_local.view(v_local.size()[0], 1, -1)        # [B, 1, 200]

        # entity sampled by strong candidates (no negative sampling)
        candidates = self.entity_embedder(candidates)    # first element is true wiki id
        scores_text = torch.matmul(candidates, v_local.view(v_local.shape[0], -1, 1) )  # B x C x dim  By B x dim x 1 (matmul) => B x C x 1
        scores_text = torch.squeeze(scores_text)    # [B, C]

        # type
        if self.type_embedder:
            types_embded = self.type_embedder(types)            # [B, T] => [B, T, 200]
            true_entity = candidates[:, 0, :].view(candidates.shape[0], 1, -1)       #[B, 1, 200]
            if self.loss_etype:
                scores_etypes = torch.matmul(true_entity, types_embded.view(types_embded.shape[0], self.encoded_dims, -1)) # [B, 1, T]
                scores_etypes = torch.squeeze(scores_etypes)            # [ B, T]
                loss_etypes = self.loss_etype(scores_etypes, types.float())     # [ B, T]
            else:
                loss_etypes = 0.0

            if self.loss_mtype:
                scores_mtypes = torch.matmul(v_local, types_embded.view(types_embded.shape[0], self.encoded_dims, -1))
                scores_mtypes = torch.squeeze(scores_mtypes)
                loss_mtypes = self.loss_mtype(scores_mtypes, types.float())
            else:
                loss_mtypes = 0.0

        # pass B x C (for each candidate score) and B x 1 true targets
        loss = self.loss_text(scores_text, targets)  + loss_etypes + loss_mtypes

        with torch.no_grad():
            for metric in self.metrics.values():
                metric(scores_text, targets)

        output_dict = {'loss': loss, "scores": scores_text} # use "scores" to compute softmax probability of posterior
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
