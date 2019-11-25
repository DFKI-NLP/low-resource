from typing import List, Dict, Any, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
import numpy as np

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("ner_extractor_pu")
class LowResourceNerExtractorPU(Model):
    """
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 ner_model: Model,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(LowResourceNerExtractorPU, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.ner_model = ner_model

        self.positive = np.eye(2)[1]
        self.negative = np.eye(2)[0]




        self.beta = 0.0
        self.gamma = 1.0

        self.loss_crf = 0.0
        self.loss_estimate = 0.0
        self.pRisk = 0.0
        self.uRisk = 0.0
        self.nRisk = 0.0

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

        self.pRisk = ner_output_dict['pRisk']
        self.uRisk = ner_output_dict['uRisk']
        self.nRisk = ner_output_dict['nRisk']
        self.loss_crf = ner_output_dict["loss_crf"]
        self.loss_estimate = ner_output_dict["loss_estimate"]
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
        metrics_dict["loss_crf"] = self.loss_crf
        metrics_dict["loss_es"] = self.loss_estimate
        metrics_dict["pRisk"] = self.pRisk
        metrics_dict["uRisk"] = self.uRisk
        metrics_dict["nRisk"] = self.nRisk

        return metrics_dict

