from typing import List, Optional

import os
import json

import spacy
from spacy.language import Language

from typing import List, Dict, Tuple, Optional

from spacy.pipeline import EntityRuler
from spacy.language import Language

# def create_entity_ruler(nlp: Language,
#                         gazetteer_patterns: Optional[List[Tuple[str, str]]] = None,
#                         gazetteer_phrases: Optional[List[Tuple[str, str]]] = None
#                         ) -> EntityRuler:
#     gazetteer_patterns = gazetteer_patterns or []
#     gazetteer_phrases = gazetteer_phrases or []
#
#     patterns = []
#     added_phrases = set()
#     for phrase, label in gazetteer_phrases:
#         if phrase.lower() not in added_phrases:
#             patterns.append({
#                     "label": label,
#                     "pattern": phrase.lower()
#             })
#             added_phrases.add(phrase.lower())
#
#     patterns += gazetteer_patterns
#
#     ruler = EntityRuler(nlp,phrase_matcher_attr="LOWER", overwrite_ents=True)
#     ruler.add_patterns(patterns)
#
#     return ruler
from src.pipeline.entity_ruler import create_entity_ruler_phrases


def _load_gaz_file(path: str) -> List[str]:
    with open(path, "r") as gaz_file:
        return [line.strip() for line in gaz_file.readlines()]


def create_nlp_pipeline(language: str = "en_core_web_sm",  # #en_core_web_lg
                        gazetteers_path: Optional[str] = None) -> Language:
    nlp = spacy.load(language)

    gazetteer_phrases = []
    if gazetteers_path is not None:
        for filename in os.listdir(gazetteers_path):
            if filename.endswith(".gaz"):
                label = os.path.splitext(filename)[0].upper()
                gazetteer_phrases += [(phrase, label) for phrase in
                                      _load_gaz_file(os.path.join(gazetteers_path,
                                                                  filename))]

        # gazetteer_patterns = []
        # if patterns_path is not None:
        #     with open(patterns_path, "r") as patterns_file:
        #         for line in patterns_file:
        #             gazetteer_patterns.append(json.loads(line))

        # if gazetteer_phrases or gazetteer_patterns:
    for name, _ in nlp.pipeline:
        nlp.remove_pipe(name)

    with nlp.disable_pipes(*[name for name, _ in nlp.pipeline]):
        entity_ruler = create_entity_ruler_phrases(nlp=nlp,
                                                   gazetteer_phrases=gazetteer_phrases)

    nlp.add_pipe(entity_ruler)

    return nlp

# GAZETTEERS_PATH = "../../data/gazetteers/"
# create_pipeline(gazetteers_path=GAZETTEERS_PATH)
