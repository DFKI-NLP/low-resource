from typing import List, Dict, Any, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("low_resource_ner_extractor")
class LowResourceNerExtractor(Model):
    """
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 ner_model: Model,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(LowResourceNerExtractor, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.ner_model = ner_model

        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                # gaz_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:

        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        # ner_input = torch.cat([embedded_text_input, gaz_tags.float()], dim=-1)

        ner_output_dict = self.ner_model(embedded_text_input=embedded_text_input,
                                         mask=mask,
                                         tags=tags,
                                         metadata=metadata)

        return ner_output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = self.ner_model.decode(output_dict)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_dict = {}
        for model_name, model in zip(["ner"], [self.ner_model]):
            model_metrics = model.get_metrics(reset)
            metrics_dict.update({
                    f"{model_name}-{metric_name}": metric
                    for metric_name, metric in model_metrics.items()
            })

        return metrics_dict
