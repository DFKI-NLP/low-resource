from typing import Dict, Tuple, Optional
import json
import logging
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, SpanField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("pc_relation_reader")
class PC_RelationDatasetReader(DatasetReader):
    """
    Reads a JSONL file containing examples from the SemEval 2010 Task 8 dataset,
    and creates a dataset suitable for relation classification.
    The JSONL could have other fields, too, but they are ignored.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        text: ``TextField``
        head: ``SpanField``
        tail: ``SpanField``
        label: ``LabelField``
    Parameters
    ----------
    max_len : ``int``
        Limit the number of tokens for each text. This is important for computing the relative offset
        of head and tail entities. (TODO: maybe there's a better way to handle this)
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the text into words or other kinds of tokens.
        Defaults to ``WordTokenizer(word_splitter=JustSpacesWordSplitter())``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(
            self,
            max_len: int,
            lazy: bool = False,
            tokenizer: Tokenizer = None,
            token_min_padding_length: int =None,
            token_indexers: Dict[str, TokenIndexer] = None,
    ) -> None:
        super().__init__(lazy)
        # nlp = spacy.load("en_core_web_sm")
        self._max_len = max_len
        self._token_min_padding_length = token_min_padding_length
        self._tokenizer = tokenizer or WordTokenizer(
            word_splitter=JustSpacesWordSplitter()
        )
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info(
                "Reading RE instances from relation_labeling_single_in_sentence.ipynb ouputs at: %s",
                file_path,
            )
            for line in data_file:
                example = json.loads(line)

                text = example["sent"]
                relation = example["label"]

                entity1, entity2 = example["entities"]

                id_ = example["id"]

                head = (entity1[0], entity1[1])
                tail = (entity2[0], entity2[1])

                if text[-1] == ".": text = text[:-1]

                yield self.text_to_instance(text, head, tail, id_, relation)

    @overrides
    def text_to_instance(
            self,
            text: str,
            head: Tuple[int, int],
            tail: Tuple[int, int],
            id_: Optional[str] = None,
            relation: Optional[str] = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        head_start, head_end = head
        tail_start, tail_end = tail

        tokenized_text = self._tokenizer.tokenize(text)
        tokenized_text = tokenized_text[: self._max_len]

        head_start = min(head_start, self._max_len - 1)
        head_end = min(head_end, self._max_len - 1)
        tail_start = min(tail_start, self._max_len - 1)
        tail_end = min(tail_end, self._max_len - 1)

        text_tokens_field = TextField(tokenized_text, self._token_indexers)
        # SpanField expects an inclusive end index

        fields = {
            "text": text_tokens_field,
            "head": SpanField(head_start, head_end, sequence_field=text_tokens_field),
            "tail": SpanField(tail_start, tail_end, sequence_field=text_tokens_field),
        }

        if id_ is not None:
            fields["metadata"] = MetadataField({"id": id_})

        if relation is not None:
            fields["label"] = LabelField(str(relation))

        return Instance(fields)

