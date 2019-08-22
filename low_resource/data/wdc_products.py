from typing import List, Dict, Any

import json

from tqdm import tqdm


def load_wdc_products(path: str) -> List[Dict[str, Any]]:
    examples = []
    with open(path, "r") as dataset_f:
        for line in tqdm(dataset_f):
            examples.append(json.loads(line))

    return examples
