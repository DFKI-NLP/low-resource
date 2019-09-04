from typing import Dict, List, Sequence, Optional
import json
import logging

from allennlp.data import Field
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (LabelField, TextField, SpanField, SequenceLabelField,
                                  MetadataField, AdjacencyField)
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("jsonl")
class JsonlDatasetReader(DatasetReader):
    """
    Reads a JSON file containing examples in JSONL format and creates a
    dataset suitable for NER. The expected keys per
    example are "token", "ner" and "pos".
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_tags``, ``chunk_tags``, ``ner_tags``.
        If you want to use one of the tags as a `feature` in your model, it should be
        specified here.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL 2003 NER data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    label_namespace: ``str``, optional (default=``labels``)
        Specifies the namespace for the chosen ``tag_label``.
    """
    _VALID_LABELS = {'ner', 'pos', 'chunk'}

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 coding_scheme: str = "IOB1",
                 label_namespace: str = "labels") -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in self._VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))
        if coding_scheme not in ("IOB1", "BIOUL"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        self._original_coding_scheme = "IOB1"

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading JSONL instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                data = json.loads(line)
                tokens = data["token"]
                ner = data["ner"]
                id_ = data.get("id")
                pos = data.get("pos")

                yield self.text_to_instance(tokens, id_, ner, pos)

    @overrides
    def text_to_instance(self,
                         text: List[str],
                         id_: Optional[str] = None,
                         ner_tags: Optional[List[str]] = None,
                         pos_tags: Optional[List[str]] = None,
                         chunk_tags: List[str] = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        tokenized_text = [Token(t) for t in text]
        sequence = TextField(tokenized_text, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokenized_text]})

        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL":
            coded_chunks = to_bioul(chunk_tags,
                                    encoding=self._original_coding_scheme) if chunk_tags is not None else None
            coded_ner = to_bioul(ner_tags,
                                 encoding=self._original_coding_scheme) if ner_tags is not None else None
        else:
            # the default IOB1
            coded_chunks = chunk_tags
            coded_ner = ner_tags

        # Add "feature labels" to instance
        if 'pos' in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError("Dataset reader was specified to use pos_tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['pos_tags'] = SequenceLabelField(pos_tags, sequence, "pos_tags")
        if 'chunk' in self.feature_labels:
            if coded_chunks is None:
                raise ConfigurationError("Dataset reader was specified to use chunk tags as "
                                         "features. Pass them to text_to_instance.")
            instance_fields['chunk_tags'] = SequenceLabelField(coded_chunks, sequence, "chunk_tags")
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError("Dataset reader was specified to use NER tags as "
                                         " features. Pass them to text_to_instance.")
            instance_fields['ner_tags'] = SequenceLabelField(coded_ner, sequence, "ner_tags")

        # Add "tag label" to instance
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence,
                                                         self.label_namespace)
        elif self.tag_label == 'pos' and pos_tags is not None:
            instance_fields['tags'] = SequenceLabelField(pos_tags, sequence,
                                                         self.label_namespace)
        elif self.tag_label == 'chunk' and coded_chunks is not None:
            instance_fields['tags'] = SequenceLabelField(coded_chunks, sequence,
                                                         self.label_namespace)

        return Instance(instance_fields)