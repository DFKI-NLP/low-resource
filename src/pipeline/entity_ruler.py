import spacy
from spacy.pipeline import EntityRuler
from spacy.language import Language
from typing import List, Dict, Tuple, Optional


def pre_defined_patterns():
    storage_pattern = [
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["gb", "mb"]}}
    ]

    weights_pattern = [
        {"LIKE_NUM": True},
        {"LOWER": {"IN": ["g", "kg", "grams", "kilograms", "lb", "lbs", "pounds", "-pound"]}}
    ]

    dimension_pattern = [
        {"LIKE_NUM": True},
        {"LOWER": "x"},
        {"LIKE_NUM": True},
        {"LOWER": "x"},
        {"LIKE_NUM": True}
    ]

    speed_pattern = [{"LIKE_NUM": True}, {"ORTH": {"IN": ["MHz", "GHz"]}}]

    write_speed_pattern = [{"LIKE_NUM": True}, {"LOWER": "Kilobytes"}, {"LOWER": "per"},
                           {"LOWER": {"IN": ["second", "second*"]}}]

    inches_pattern = [{"LIKE_NUM": True}, {"LOWER": {"IN": ["by", "x"]}}, {"LIKE_NUM": True},
                      {"LOWER": {"IN": ["inches", "maximum"]}}]

    size_pattern = [{"LIKE_NUM": True}, {"LOWER":  {"IN": ["inches", "-inche", "inche"]}}]
    time_span_pattern = [{"LIKE_NUM": True}, {"LOWER": "to"}, {"LIKE_NUM": True}, {"LOWER": "hours"}]

    electic_current_pattern = [{"IS_DIGIT": True}, {"LOWER": {"REGEX": "m?A$"}}]

    patterns = [
        {"label": "UNIT_DATA_STORAGE", "pattern": storage_pattern},
        {"label": "DIMENSIONS", "pattern": dimension_pattern},
        {"label": "UNIT_WEIGHTS", "pattern": weights_pattern},
        {"label": "UNIT_SPEED", "pattern": speed_pattern},
        {"label": "UNIT_WRITESPEED", "pattern": write_speed_pattern},
        {"label": "UNIT_RESOLUTION", "pattern": inches_pattern},
        {"label": "UNIT_ELECTIC_CURRENT", "pattern": electic_current_pattern},
        {"label": "TIME_SPAN", "pattern": time_span_pattern},
        {"label": "SIZE", "pattern": size_pattern}
    ]


    return patterns

    # nlp = spacy.load("en_core_web_sm")  # en_core_web_lg #en_core_web_sm
    #
    # ruler = EntityRuler(nlp, overwrite_ents=True)  # phrase_matcher_attr="LOWER",
    # ruler.add_patterns(patterns)
    #
    # ruler.to_disk("data/ruler/patterns.jsonl")


def create_entity_ruler_phrases(nlp: Language,
                                gazetteer_phrases: Optional[List[Tuple[str, str]]] = None
                                ) -> EntityRuler:
    gazetteer_phrases = gazetteer_phrases or []

    patterns = []
    added_phrases = set()
    for phrase, label in gazetteer_phrases:
        if phrase.lower() not in added_phrases:
            if " " in phrase:
                p_list = phrase.split(" ")
                sub_patterns = [{"LOWER": word.lower()} for word in p_list]
                patterns.append({
                    "label": label,
                    "pattern": sub_patterns
                })
            else:
                patterns.append({
                    "label": label,
                    "pattern": [{"LOWER": phrase.lower()}]
                })
            added_phrases.add(phrase.lower())

    patterns += pre_defined_patterns()

    ruler = EntityRuler(nlp, overwrite_ents=True)  # phrase_matcher_attr="LOWER",
    ruler.add_patterns(patterns)

    return ruler
