#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with an ElasticSearch index"""

import logging
import re
from abc import abstractmethod
from functools import partial
from multiprocessing.pool import ThreadPool

from elasticsearch import Elasticsearch

from drka.retriever.base_ranker import BaseRanker
from . import DEFAULTS
from . import utils

logger = logging.getLogger(__name__)


class ElasticDocRanker(BaseRanker):
    """ Connect to an ElasticSearch index.
        Score pairs based on ElasticSearch
    """

    def __init__(self, elastic_url=None, elastic_index=None, elastic_fields=None, elastic_fields_weights=None,
                 elastic_field_doc_name=None, strict=True, elastic_field_content=None, elastic_field_metadata=None,
                 auth=None):
        """
        Args:
            elastic_url: URL of the ElasticSearch server containing port
            elastic_index: Index name of ElasticSearch
            elastic_fields: Fields of the ElasticSearch index to search in
            elastic_field_doc_name: Field containing the name of the document (index)
            strict: fail on empty queries or continue (and return empty result)
            elastic_field_content: Field containing the content of document in plain text
        """
        # Load from disk
        elastic_url = elastic_url or DEFAULTS['elastic_url']
        logger.info('Connecting to %s' % elastic_url)

        super().__init__("elastic")

        self.es = Elasticsearch(hosts=elastic_url, http_auth=auth) if auth else Elasticsearch(hosts=elastic_url)
        self.elastic_index = elastic_index
        self.elastic_fields = elastic_fields
        self.elastic_fields_weights = elastic_fields_weights
        self.elastic_field_doc_name = elastic_field_doc_name
        self.elastic_field_content = elastic_field_content
        self.elastic_field_metadata = elastic_field_metadata
        self.strict = strict
        self.filter_most_relevant = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # Elastic Ranker

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        field_index = self.elastic_field_doc_name
        if isinstance(field_index, list):
            field_index = '.'.join(field_index)
        result = self.es.search(index=self.elastic_index, body={'query': {'match': {field_index: doc_id}}})
        return result['hits']['hits'][0]['_id']

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        result = self.es.search(index=self.elastic_index, body={'query': {'match': {"_id": doc_index}}})
        source = result['hits']['hits'][0]['_source']
        return utils.get_field(source, self.elastic_field_doc_name)

    def closest_docs(self, query, k=1, **kwargs):
        """Closest docs by using ElasticSearch
        """
        del kwargs

        results = self.es.search(index=self.elastic_index, body={
            'size': k,
            'query':
                {
                    'multi_match': {
                        'query': query,
                        'type': 'most_fields',
                        'fields': self.elastic_fields}
                }
        })

        hits = results['hits']['hits']

        doc_ids = [utils.get_field(row['_source'], self.elastic_field_doc_name) for row in hits]
        doc_scores = [row['_score'] for row in hits]
        return doc_ids, doc_scores, 0

    def closest_vector(self, vector, k=1, tags="em", **kwargs):
        results = self.es.search(index=self.elastic_index, body={
            'query': {
                "match_all": {}
            }
        })
        vector = [float(entry) for entry in vector.replace("[", "").replace("]", "").split(",")]
        for result in results['hits']['hits']:
            print(result["_source"]["doc2vec"])
            print(vector)
            print(type(result["_source"]["doc2vec"][0]))
            print([sum(ai * bi for ai, bi in zip(a, b)) for a, b in zip(result["_source"]["doc2vec"], vector)])

    def closest_docs_text(self, query, k=1, tags="em", visitor=None, **kwargs):
        """Closest docs and content by using ElasticSearch
        """
        del kwargs

        _body = {
            'size': k,
            'query': {
                'multi_match': {
                    'query': query,
                    'type': 'most_fields',
                    'fields': self.elastic_fields,
                    "slop": 1000
                }
            },
            "highlight": {
                "fields": {
                    self.elastic_field_content: {}
                },
                # TODO: Move these tags to the frontend or use a visitor
                "pre_tags": "<" + tags + ">",
                "post_tags": "</" + tags + ">"
            }
        }
        _body = visitor.process(self, query, k) if visitor else _body

        results = self.es.search(index=self.elastic_index, body=_body)

        if results and "hits" in results and "hits" in results['hits']:
            hits_ = results['hits']['hits']
        else:
            hits_ = []

        return {"answers": hits_}

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multi-threaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    # Elastic DB

    def close(self):
        """Close the connection to the database."""
        self.es = None

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        results = self.es.search(index=self.elastic_index, body={"query": {"match_all": {}}})
        _hits = results['hits']['hits']
        doc_ids = [utils.get_field(result['_source'], self.elastic_field_doc_name) for result in _hits]
        return doc_ids

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        idx = self.get_doc_index(doc_id)
        result = self.es.get(index=self.elastic_index, doc_type='_doc', id=idx)

        if result is not None:
            text_result = result['_source'][self.elastic_field_content]
            # HACK: some extracted text does not contain sentences/blocks that end with period
            #       this breaks the QA functionality and slows down the process
            text_result = text_result.replace("\n \n", ".\n")
            text_result = re.sub(r'([^.][ \t]*)\n([ \t]*[A-Z])', r'\1.\n\2', text_result)

            text_result = re.sub(r'([a-z]+)\?([A-Z])', r'\1?\n\2', text_result)
            text_result = text_result.replace(" ", "").replace(" ", "").replace(" ■", "; ")

            if "title" in result['_source']:
                text_result = result['_source']["title"].replace("-", " : ") + " " + text_result

            text_result = text_result.replace("i.e. ", "such as ")
            ####

        return text_result


class ElasticVisitor:

    @abstractmethod
    def process(self, ranker: ElasticDocRanker, query: str, k: int):
        pass
