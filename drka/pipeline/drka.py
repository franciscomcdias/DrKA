#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Full DrKA pipeline."""

import heapq
import logging
import time
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

import math
import regex
import torch

from . import DEFAULTS
from .. import SER_MODEL_EXTENSION
from .. import reader
from .. import tokenizers
from ..reader.data import ReaderDataset, SortedBatchSampler
from ..reader.vector import batchify

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Multiprocessing functions to fetch and tokenize text
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None
PROCESS_CANDS = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Main DrKA pipeline
# ------------------------------------------------------------------------------


class DrKA(object):
    # Target size for squashing short paragraphs together.
    # 0 = read every paragraph independently
    # infty = read all paragraphs together
    GROUP_LENGTH = 0

    def __init__(
            self,
            reader_model=None,
            embedding_file=None,
            tokenizer=None,
            fixed_candidates=None,
            batch_size=128,
            cuda=True,
            data_parallel=False,
            max_loaders=5,
            num_workers=None,
            db_config=None,
            ranker_config=None
    ):
        """Initialize the pipeline.

        Args:
            reader_model: model file from which to load the DocReader.
            embedding_file: if given, will expand DocReader dictionary to use
              all available pre-trained embeddings.
            tokenizer: string option to specify tokenizer used on docs.
            fixed_candidates: if given, all predictions will be constrained to
              the set of candidates contained in the file. One entry per line.
            batch_size: batch size when processing paragraphs.
            cuda: whether to use the gpu.
            data_parallel: whether to use multiple gpus.
            max_loaders: max number of async data loading workers when reading.
              (default is fine).
            num_workers: number of parallel CPU processes to use for tokenizing
              and post processing resuls.
            db_config: config for doc db.
            ranker_config: config for ranker.
        """
        self.batch_size = batch_size
        self.max_loaders = max_loaders
        self.fixed_candidates = fixed_candidates is not None
        self.cuda = cuda

        logger.info("Initializing document ranker...")
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get("class", DEFAULTS["ranker"])
        ranker_opts = ranker_config.get("options", {})
        self.ranker = ranker_class(**ranker_opts)

        logger.info("Initializing document reader...")
        reader_model = reader_model or DEFAULTS["reader_model"]
        self.reader = reader.DocReader.load(reader_model, normalize=False)
        if embedding_file:
            logger.info("Expanding dictionary...")
            if embedding_file.endswith(SER_MODEL_EXTENSION):
                self.reader.load_serialized_embeddings(embedding_file)
            else:
                words = reader.utils.index_embedding_words(embedding_file)
                added = self.reader.expand_dictionary(words)
                self.reader.load_embeddings(added, embedding_file)
        if cuda:
            self.reader.cuda()
        if data_parallel:
            self.reader.parallelize()

        if not tokenizer:
            tok_class = DEFAULTS["tokenizer"]
        else:
            tok_class = tokenizers.get_class(tokenizer)
        annotators = tokenizers.get_annotators_for_model(self.reader)
        tok_opts = {"annotators": annotators}

        logger.info("Loading ranker: " + self.ranker.name)

        # ElasticSearch / Custom are also used as backends if used as ranker
        if self.ranker.name in ["elastic", "custom"]:

            db_class = ranker_class
            db_opts = ranker_opts
            self.num_workers = 1

        else:

            db_config = db_config or {}
            db_class = db_config.get("class", DEFAULTS["db"])
            db_opts = db_config.get("options", {})

            logger.info("Initializing tokenizer and document retrievers...")
            self.num_workers = num_workers

        self.processes = ProcessPool(
            self.num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, db_class, db_opts, fixed_candidates)
        )

    def _split_doc(self, doc, ranker, queries, data):
        """Given a doc, split it into chunks (by paragraph)."""

        if hasattr(ranker, "split_doc"):

            for split in ranker.split_doc(doc, queries[0], data):
                yield split

        else:

            current, current_len = [], 0

            for split in regex.split(r'\n+', doc):
                split = split.strip()
                if len(split) == 0:
                    continue
                # Maybe group paragraphs together until we hit a length limit
                if len(current) > 0 and current_len + len(split) > self.GROUP_LENGTH:
                    yield " ".join(current)
                    current = []
                    current_len = 0
                current.append(split)
                current_len += len(split)

            if len(current) > 0:
                yield " ".join(current)

    def _get_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = ReaderDataset(data, self.reader)
        sampler = SortedBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=batchify,
            pin_memory=self.cuda,
        )
        return loader

    def process(self, query, candidates=None, top_n=1, n_docs=5, context=None, data=None):

        """Run a single query."""
        predictions = self.process_batch(
            [query], [candidates] if candidates else None,
            top_n, n_docs, context,
            data
        )
        return predictions[0]

    def process_batch(self, queries, candidates=None, top_n=1, n_docs=5, context=None, data=None):
        """Run a batch of queries (more efficient)."""

        def get_page_numbers(all_page_numbers_, query_number_, rel_didx_):
            if len(all_page_numbers_) >= query_number_ and all_page_numbers_[query_number_]:
                return all_page_numbers_[query_number_][rel_didx_]
            else:
                return 0

        t0 = time.time()
        logger.info("Processing %d queries..." % len(queries))
        logger.info("Retrieving top %d docs..." % n_docs)

        if not context:
            context = {"return": False, "window": None}

        # Rank documents for queries.
        if len(queries) == 1:

            id_token = data["id_token"] if data and "id_token" in data else ""
            access_token = data["access_token"] if data and "access_token" in data else ""

            if id_token or access_token:
                ranked = [self.ranker.closest_docs(queries[0], k=n_docs, id_token=id_token, access_token=access_token)]
            else:
                ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.ranker.batch_closest_docs(
                queries, k=n_docs, num_workers=self.num_workers
            )

        all_doc_ids, all_doc_scores, all_page_numbers = zip(*ranked)

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_doc_ids = list({d for doc_ids in all_doc_ids for d in doc_ids})
        doc_id2doc_index = {doc_id: doc_index for doc_index, doc_id in enumerate(flat_doc_ids)}

        if self.ranker.name in ["elastic", "custom"]:
            # HACK, as cannot pickle thread-locked objects
            doc_texts = [self.ranker.get_doc_text(flat_docid) for flat_docid in flat_doc_ids]
        else:
            doc_texts = self.processes.map(fetch_text, flat_doc_ids)


        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        flat_splits = []
        doc_index2split_index = []
        for text in doc_texts:

            doc_index2split_index.append([len(flat_splits), -1])
            for split in self._split_doc(text, self.ranker, queries, data):
                flat_splits.append(split)
            doc_index2split_index[-1][1] = len(flat_splits)

        # Push through the tokenizers as fast as possible.
        q_tokens = self.processes.map_async(tokenize_text, queries)
        s_tokens = self.processes.map_async(tokenize_text, flat_splits)
        q_tokens = q_tokens.get()
        s_tokens = s_tokens.get()

        # Group into structured example inputs. Examples' ids represent
        # mappings to their question, document, and split ids.
        examples = []
        for query_number in range(len(queries)):
            for rel_didx, document_id in enumerate(all_doc_ids[query_number]):

                start, end = doc_index2split_index[doc_id2doc_index[document_id]]

                for sidx in range(start, end):
                    if len(q_tokens[query_number].words()) > 0 and len(s_tokens[sidx].words()) > 0:
                        examples.append({
                            "id": (query_number, rel_didx, sidx),
                            "question": q_tokens[query_number].words(),
                            "qlemma": q_tokens[query_number].lemmas(),
                            "document": s_tokens[sidx].words(),
                            "lemma": s_tokens[sidx].lemmas(),
                            "pos": s_tokens[sidx].pos(),
                            "ner": s_tokens[sidx].entities(),
                        })

        logger.info("Reading %d paragraphs..." % len(examples))

        # Push all examples through the document reader.
        # We decode argmax start/end indices asychronously on CPU.
        result_handles = []
        num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        for batch in self._get_loader(examples, num_loaders):
            if candidates or self.fixed_candidates:
                batch_cands = []
                for ex_id in batch[-1]:
                    batch_cands.append({
                        "input": s_tokens[ex_id[2]],
                        "cands": candidates[ex_id[0]] if candidates else None
                    })
                handle = self.reader.predict(
                    batch, batch_cands, async_pool=self.processes
                )
            else:
                handle = self.reader.predict(batch, async_pool=self.processes)
            result_handles.append((handle, batch[-1], batch[0].size(0)))

        # Iterate through the predictions, and maintain priority queues for
        #         # top scored answers for each question in the batch.
        queues = [[] for _ in range(len(queries))]
        for result, ex_ids, batch_size in result_handles:
            s, e, score = result.get()
            for i in range(batch_size):
                # We take the top prediction per split.
                if len(score[i]) > 0:
                    item = (score[i][0], ex_ids[i], s[i][0], e[i][0])
                    queue = queues[ex_ids[i][0]]
                    if len(queue) < top_n:
                        heapq.heappush(queue, item)
                    else:
                        heapq.heappushpop(queue, item)

        # Arrange final top prediction data.
        all_predictions = []

        for queue in queues:
            predictions = []
            while len(queue) > 0:
                score, (query_number, rel_didx, sidx), s, e = heapq.heappop(queue)
                prediction = {
                    "doc_id": all_doc_ids[query_number][rel_didx],
                    "page_number": get_page_numbers(all_page_numbers, query_number, rel_didx),
                    "span": s_tokens[sidx].slice(s, e + 1).untokenize(),
                    "doc_score": float(all_doc_scores[query_number][rel_didx]),
                    "span_score": float(score),
                    "question": queries[0]
                }
                if context["return"]:
                    prediction["context"] = {
                        "text": s_tokens[sidx].untokenize(),
                        "start": s_tokens[sidx].offsets()[s][0],
                        "end": s_tokens[sidx].offsets()[e][1],
                    }
                predictions.append(prediction)
            all_predictions.append(predictions[-1::-1])

        logger.info("Processed %d queries in %.4f (s)" %
                    (len(queries), time.time() - t0))

        return all_predictions
