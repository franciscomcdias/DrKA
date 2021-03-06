#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from drka import DATA_DIR

DEFAULTS = {
    'db_path': os.path.join(DATA_DIR, 'data/docs.db'),
    'tfidf_path': os.path.join(DATA_DIR, 'data/docs-tfidf.npz'),
    'elastic_url': 'http://localhost:9200'
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .base_ranker import BaseRanker
from .doc_db_ranker import DocDB
from .elastic_doc_ranker import ElasticDocRanker
from .tfidf_doc_ranker import TfidfDocRanker


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'sql':
        return DocDB
    if name == 'elastic':
        return ElasticDocRanker
    raise RuntimeError('Invalid retriever class: %s' % name)
