# Low-resource Lenovo

## Task
Extract products of a specific category (e.g., washing machines) with corresponding attributes (e.g., brand, efficiency class, features) and their relation.

## Installation

First, clone the repository to your machine and install the requirements with the following command:

```bash
pip install -r requirements.txt
```

## Data

### Datasets
- Amazon Reviews Metadata [[Description]](http://jmcauley.ucsd.edu/data/amazon/links.html) [[Download]](http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz)
- WDC Product Corpus[[Description]](http://webdatacommons.org/largescaleproductcorpus/) [[Download]](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/offers_english.json.gz)

### Derived Datasets
The datasets created by filtering the original datasets can be found in `data/datasets`.

- `amazon_metadata_washer.json` was created by filtering the Amazon Reviews Metadata by category `Appliances | Washers & Dryers`
- `wdc_washing_machines.jsonl` was created by filtering the WDC Product Corpus by product names containing `washing machine`.

### Gazetteers
The gazetteers can be found in `data/gazetteers`.

- `brands.gaz` contains brands of known washing machine producers.
- `energy_efficiency.gaz` contains energy efficiency classes.
- `patterns.jsonl` contains patterns to identify attributes, such as unit of mass and unit of energy. 

### Distant Supervision
`train_distant.json` contains the distantly supervised dataset created from `amazon_metadata_washer.json` and `wdc_washing_machines.jsonl`. It is created by running the code in `notebooks/create_dataset.ipynb`
