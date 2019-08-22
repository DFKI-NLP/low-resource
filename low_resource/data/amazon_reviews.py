from typing import List, Dict, Any

import ast

from tqdm import tqdm


def load_amazon_metadata(path: str) -> List[Dict[str, Any]]:
    examples = []
    with open(path, "r") as dataset_f:
        for line in tqdm(dataset_f):
            examples.append(ast.literal_eval(line))

    return examples
