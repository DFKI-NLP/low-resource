from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

import json


@Predictor.register('low-resource-tagger-predictor-apt')
class SentenceTaggerPredictorApt(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    # @overrides
    # def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
    #     instances = self._batch_json_to_instances(inputs)
    #     return self.predict_batch_instance(instances)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """

        return self._dataset_reader.text_to_instance(json_dict["token"], json_dict["id"], json_dict["ner"],
                                                     json_dict["pos"])

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        words = outputs["words"]
        predicted_tags = outputs["tags"]
        ners = outputs["label_tags"]
        mask = outputs["mask"]

        ents = []
        i = 0
        total_len = 0
        seq_len = len(ners)
        while i < len(predicted_tags):
            tag_pred = predicted_tags[i]
            tag = ners[i]

            if str(tag_pred) != "O":  # and str(tag) == "0"

                ent_len = len(str(words[i]))

                j = i + 1
                while j < seq_len:
                    if predicted_tags[j] == 'O':  # here we merge the entities
                        break
                    ent_len += len(words[j])
                    j += 1

                if j > i + 1 and words[i] == words[i + 1]:
                    pass
                else:
                    ents.append({"name": " ".join(words[i:j]),
                                 "label": str(tag_pred)})
                i = j
                total_len += ent_len
            else:
                total_len += len(str(words[i]))
                i += 1

        if len(ents) > 0:
            data_predict = {"ents": ents}
            return json.dumps(data_predict) + "\n"
        else:
            return ""
