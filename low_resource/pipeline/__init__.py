from typing import List, Optional

import os
import json

import spacy
from spacy.language import Language
from low_resource.pipeline.entity_recognition import create_entity_ruler


def _load_gaz_file(path: str) -> List[str]:
    with open(path, "r") as gaz_file:
        return [line.strip() for line in gaz_file.readlines()]


def create_pipeline(language: str = "en_core_web_lg",
                    patterns_path: Optional[str] = None,
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

    gazetteer_patterns = []
    if patterns_path is not None:
        with open(patterns_path, "r") as patterns_file:
            for line in patterns_file:
                gazetteer_patterns.append(json.loads(line))

    if gazetteer_phrases or gazetteer_patterns:
        with nlp.disable_pipes(*[name for name, _ in nlp.pipeline]):
            entity_ruler = create_entity_ruler(nlp=nlp,
                                               gazetteer_patterns=gazetteer_patterns,
                                               gazetteer_phrases=gazetteer_phrases)

        nlp.add_pipe(entity_ruler)

    return nlp
