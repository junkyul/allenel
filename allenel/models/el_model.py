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
                 type_embedder: Embedding = None,
                 word_embedding_dropout: float = 0.4,
                 coherence_embedding_dropout: float = 0.6,
                 left_seq2vec: Seq2VecEncoder = None,
                 right_seq2vec: Seq2VecEncoder = None,
                 ff_seq2vecs : FeedForward = None,
                 ff_context: FeedForward = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
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

        self.coherence_embedder = coherence_embedder
        self.coherence_dropout = torch.nn.Dropout(coherence_embedding_dropout)
        self.coherence_embedder_relu = torch.nn.ReLU(inplace=False)

        self.ff_context = ff_context

        self.entity_embedder = entity_embedder

        self.type_embedder = type_embedder
        if self.type_embedder:
            self.type_embedder_sigmoid = torch.nn.Soft

        # loss
        self.loss_context = torch.nn.CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
        }
        initializer(self)

    @overrides
    def forward(self,
                #sentence: Dict[str, torch.LongTensor],             # shape B x seq_len x 300
                sentence_left: Dict[str, torch.LongTensor],
                sentence_right: Dict[str, torch.LongTensor],
                mention: List[str],
                candidates: Dict[str, torch.LongTensor],
                mention_normalized: List[str],
                types: Dict[str, torch.LongTensor],
                coherences: Dict[str, torch.LongTensor],
                targets: torch.LongTensor = None,
                # wid: List[str] = None,
                # title: List[str] = None,
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
        local_context = self.ff_seq2vecs(sentence_encoded)

        # coherence
        coherences = self.coherence_embedder(coherences)
        coherences = torch.sum(coherences, dim=1).view(coherences.size()[0], -1)        # B x
        coherences = self.coherence_dropout(coherences)
        coherences = self.coherence_embedder_relu(coherences)

        # context
        v_local = torch.cat( (local_context, coherences), dim=1)
        v_local = self.ff_context(v_local)
        v_local = v_local.view(v_local.size()[0], 1, -1)

        if self.type_embedder:
            pass

        # entity sampled by candidates
        candidates = self.entity_embedder(candidates)    # first element is true label

        scores = torch.matmul(candidates, v_local.view(v_local.size()[0], -1, 1) )
        scores = torch.squeeze(scores)
        loss = self.loss_context(scores, targets)

        with torch.no_grad():
            for metric in self.metrics.values():
                metric(scores, targets)

        output_dict = {'loss': loss, "scores": scores}       # use "scores" to compute softmax probability
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
