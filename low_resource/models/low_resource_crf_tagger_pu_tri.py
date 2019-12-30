from typing import Dict, Optional, List, Any

from allennlp.data.dataset_readers.dataset_utils.ontonotes import TypedStringSpan
from allennlp.data.dataset_readers.dataset_utils.span_utils import InvalidTagSequence
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn as nn

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from typing import Callable, List, Set, Tuple, TypeVar, Optional
import warnings

import torchvision.utils as vutils

from low_resource.metrics import F1Measure


@Model.register("low_resource_crf_tagger_pu_tri")
class LowResourceCrfTaggerPU3(Model):
    """
    The ``CrfTagger`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the tokens ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : ``str``, optional (default=``labels``)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : ``FeedForward``, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if ``calculate_span_f1`` or ``constrain_crf_decoding`` is true.
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : ``bool``, optional (default=``None``)
        If ``True``, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    calculate_span_f1 : ``bool``, optional (default=``None``)
        Calculate span-level F1 metrics during training. If this is ``True``, then
        ``label_encoding`` is required. If ``None`` and
        label_encoding is specified, this is set to ``True``.
        If ``None`` and label_encoding is not specified, it defaults
        to ``False``.
    dropout:  ``float``, optional (default=``None``)
    verbose_metrics : ``bool``, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 label_namespace: str = "labels",
                 feedforward: Optional[FeedForward] = None,
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 constrain_crf_decoding: bool = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 prior: Optional[float] = 0.05,
                 prior_I: Optional[float] = 0.03,
                 gamma: Optional[float] = 0.5,
                 m: Optional[float] = 0.6,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.label_namespace = label_namespace
        # self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim,
                                                           self.num_tags))

        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.")
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            transition_label_encoding = "BIO" if label_encoding == "BO" else label_encoding
            constraints = allowed_transitions(transition_label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
            self.num_tags, constraints,
            include_start_end_transitions=include_start_end_transitions
        )
        self.entroyLoss = nn.CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            # "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.calculate_span_f1 = calculate_span_f1

        self.prior = prior
        self.prior_I = prior_I
        self.beta = 0.0
        self.gamma = gamma
        self.m = m

        self.loss_crf = 0.0
        self.loss_estimate = 0.0
        self.pRisk = 0.0
        self.uRisk = 0.0
        self.nRisk = 0.0

        if not label_encoding:
            raise ConfigurationError("calculate_span_f1 is True, but "
                                     "no label_encoding was specified.")
        if (not calculate_span_f1):
            self._f1_metric = F1Measure(vocabulary=self.vocab)
        else:
            if label_encoding == "BO":
                self._f1_metric = SpanBasedF1Measure(vocab,
                                                     tag_namespace=label_namespace,
                                                     label_encoding=None,
                                                     tags_to_spans_function=self.binary_tags_to_spans)
            else:
                self._f1_metric = SpanBasedF1Measure(vocab,
                                                     tag_namespace=label_namespace,
                                                     label_encoding=label_encoding)

        # check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
        #                        "text field embedding dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")

        initializer(self)

    def binary_tags_to_spans(self, tag_sequence: List[str],
                             classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
        """
            Returns
            -------
            spans : List[TypedStringSpan]
                The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
                Note that the label `does not` contain any BIO tag prefixes.
            """

        spans: Set[Tuple[str, Tuple[int, int]]] = set()
        seq_len = len(tag_sequence)
        i = 0
        while i < seq_len:
            tag = tag_sequence[i][0]
            conll_tag = tag_sequence[i][2:]
            if tag == "B":
                j = i + 1
                while j < seq_len:
                    if tag_sequence[j][0] != 'B':
                        break
                    j += 1
                span_start, span_end = i, j - 1  # exclusive
                spans.add((conll_tag, (span_start, span_end)))
                i = j + 1
            else:
                i += 1
            # Last token might have been a part of a valid span.
            if tag_sequence[seq_len - 1][0] == "B":
                conll_tag = tag_sequence[seq_len - 1][2:]
                spans.add((conll_tag, (seq_len - 1, seq_len - 1)))

        return list(spans)

    @overrides
    def forward(self,  # type: ignore
                # tokens: Dict[str, torch.LongTensor],
                embedded_text_input: torch.FloatTensor,
                mask: torch.FloatTensor,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : ``Dict[str, torch.LongTensor]``, required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : ``torch.LongTensor``, optional (default = ``None``)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence to be tagged under a 'words' key.
        Returns
        -------
        An output dictionary consisting of:
        logits : ``torch.FloatTensor``
            The logits that are the output of the ``tag_projection_layer``
        mask : ``torch.LongTensor``
            The text field mask for the input tokens
        tags : ``List[List[int]]``
            The predicted tags using the Viterbi algorithm.
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised. Only computed if gold label ``tags`` are provided.
        """
        # embedded_text_input = self.text_field_embedder(tokens)
        # mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)
        best_paths = self.crf.viterbi_tags(logits, mask)

        # Just get the tags and ignore the score.
        predicted_tags = [x for x, y in best_paths]

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        device = logits.get_device()
        logits_dim = logits.size()[-1]
        if tags is not None:

            # Add negative log-likelihood as loss
            pRisk = - self.crf(logits, tags, mask)

            gt__sum = tags.sum(1).int().gt(0).sum()
            if gt__sum == 0:
                uRisk = torch.FloatTensor([0]).squeeze().to(device)
                nRisk = torch.FloatTensor([0]).squeeze().to(device)
                risk = self.m * pRisk
            else:
                slice_index = tags.sum(1).gt(0).nonzero().long().squeeze()
                slice_logits = logits.index_select(0, slice_index)
                slice_tags = tags.index_select(0, slice_index)
                slice_mask = mask.index_select(0, slice_index)
                uRisk = - self.crf(slice_logits, torch.zeros(slice_tags.size()).long().to(device),
                                   (slice_tags == 0).long() * slice_mask)
                nRisk = self.prior * uRisk

                risk = self.m * pRisk - nRisk

            # if risk < self.beta:
            #     risk = -self.gamma * nRisk
            if risk < self.beta:
                risk = self.gamma * nRisk

            if not risk.grad_fn:
                risk = pRisk

            output["loss"] = risk

            self.pRisk = float(pRisk.clone().detach().item())
            self.uRisk = float(uRisk.clone().detach().item())
            self.nRisk = float(nRisk.clone().detach().item())
            self.loss_estimate = float(risk.clone().detach().item())
            self.loss_crf = float(pRisk.clone().detach().item())

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())

            self._f1_metric(class_probabilities, tags, mask.float())
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        output_dict["tags"] = [
            [self.vocab.get_token_from_index(tag, namespace=self.label_namespace)
             for tag in instance_tags]
            for instance_tags in output_dict["tags"]
        ]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {metric_name: metric.get_metric(reset) for
                             metric_name, metric in self.metrics.items()}

        f1_dict = self._f1_metric.get_metric(reset=reset)
        if self._verbose_metrics:
            metrics_to_return.update(f1_dict)
        else:
            metrics_to_return.update({
                x: y for x, y in f1_dict.items() if
                "overall" in x})

        metrics_to_return["pRisk"] = self.pRisk
        metrics_to_return["uRisk"] = self.uRisk
        metrics_to_return["nRisk"] = self.nRisk
        metrics_to_return["loss_estimate"] = self.loss_estimate
        metrics_to_return["loss_crf"] = self.loss_crf


        return metrics_to_return