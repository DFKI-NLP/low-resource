from typing import List, Dict, Tuple, Optional

from spacy.pipeline import EntityRuler
from spacy.language import Language


def create_entity_ruler(nlp: Language,
                        gazetteer_patterns: Optional[List[Tuple[str, str]]] = None,
                        gazetteer_phrases: Optional[List[Tuple[str, str]]] = None
                        ) -> EntityRuler:
    gazetteer_patterns = gazetteer_patterns or []
    gazetteer_phrases = gazetteer_phrases or []

    patterns = []
    added_phrases = set()
    for phrase, label in gazetteer_phrases:
        if phrase.lower() not in added_phrases:
            patterns.append({
                    "label": label,
                    "pattern": phrase.lower()
            })
            added_phrases.add(phrase.lower())

    patterns += gazetteer_patterns

    ruler = EntityRuler(nlp, phrase_matcher_attr="LOWER", overwrite_ents=True)
    ruler.add_patterns(patterns)
    return ruler
