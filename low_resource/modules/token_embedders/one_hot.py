import logging

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TokenEmbedder.register("one_hot")
class OneHot(TokenEmbedder):
    def __init__(self,
                 embedding_dim: int,
                 vocab_namespace: str = None) -> None:
        super(OneHot, self).__init__()
        self.embedding_dim = embedding_dim
        self._vocab_namespace = vocab_namespace

    @overrides
    def get_output_dim(self) -> int:
        return self.embedding_dim

    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ
        inputs = inputs.unsqueeze(-1)

        embedded_size = list(inputs.size())
        embedded_size[-1] = self.embedding_dim

        embedded = torch.zeros(embedded_size,
                               dtype=inputs.dtype,
                               layout=inputs.layout,
                               device=inputs.device)

        embedded.scatter_(-1, inputs, 1.)

        return embedded.float()

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'OneHot':  # type: ignore
        embedding_dim = params.pop_int('embedding_dim', None)
        vocab_namespace = params.pop("vocab_namespace", None if embedding_dim else "ner_tokens")
        if embedding_dim is None:
            embedding_dim = vocab.get_vocab_size(vocab_namespace)

        return cls(embedding_dim=embedding_dim,
                   vocab_namespace=vocab_namespace)
