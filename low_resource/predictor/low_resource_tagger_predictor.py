from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor

import json


@Predictor.register('low-resource-tagger-predictor')
class SentenceTaggerPredictor(Predictor):
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
        ners = outputs["tags"]

        ents = []
        i = 0
        j = 0
        total_len = 0
        seq_len = len(ners)
        while i < len(ners):
            tag = ners[i]

            if str(tag) == "O":
                total_len += len(str(words[i]))
                i += 1
                continue

            ent_len = len(str(words[i]))
            j = i + 1
            while j < seq_len:
                if ners[j] == 'O':
                    break
                ent_len += len(words[j])
                j += 1

            ents.append({"start": total_len+i, "end": total_len+ent_len+j,"label": str(tag).split("-")[1]})
            i = j
            total_len += ent_len

        # raws =[]
        # i=0
        # for w,t in zip(words,ners):
        #     raws.append(str(w)+"/"+str(t)+"/"+str(i))
        #     i+=1

        data_predict = {"text": " ".join(words), "ents": ents,"title":None,"settings":{}}
        return json.dumps(data_predict) + "\n"
