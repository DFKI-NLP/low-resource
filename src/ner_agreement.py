#!/usr/bin/python
# -*- coding: utf8 -*-
"""

@date: 18.03.20
@author: leonhard.hennig@dfki.de
"""
import os
from nltk.metrics.agreement import AnnotationTask

def _read_file(path, annotator, fidx):
    result = []
    with open(path) as f:
        for lidx, line in enumerate(f):
            if len(line.strip()) > 0:
                word_ner = line.strip().split(" ")
                result.append((annotator, str(fidx) + "-" + str(lidx) + "-" + word_ner[0], word_ner[1]))
    return result

basepath = '/home/leonhard/Dokumente/code/dfki/corpora/amazon-product-corpus/Lenovo_Product_Reviews_2020-03-18_1807_FullExport_CoNLL02/'
labels_elif = []
labels_ulli = []
labels_curator = []
for fidx, doc_dir in enumerate(os.listdir(basepath + "annotation/")):
    labels_elif.extend(_read_file(basepath + "annotation/" + doc_dir + "/elif.conll", "elif", fidx))
    labels_ulli.extend(_read_file(basepath + "annotation/" + doc_dir + "/ulli.conll", "ulli", fidx))
    labels_curator.extend(_read_file(basepath + "curation/" + doc_dir + "/CURATION_USER.conll", "cur", fidx))

task = AnnotationTask(data = labels_elif + labels_ulli + labels_curator)
print("Krippendorf's alpha: " + str(task.alpha()))
print("Cohen's kappa: " + str(task.kappa()))
