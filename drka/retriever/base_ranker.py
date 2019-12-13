import abc
import re

import nltk
import stop_words
from sklearn.feature_extraction.text import CountVectorizer


class BaseRanker(object):
    """
    Base class for a Ranker
    """

    def __init__(self, name, language="english"):
        self.name = name
        self.cache = {}
        self.stop_words = set(stop_words.get_stop_words(language)) if language else []
        self.filter_most_relevant = False

    def close(self):
        self.cache = {}

    @abc.abstractmethod
    def closest_docs(self, query, k=1, **kwargs):
        pass

    def get_doc_ids(self):
        """Fetch all ids of docs and stores them in the cache."""
        return list(self.cache.keys())

    def get_doc_metadata(self, doc_id=None):
        """Fetch all metadata stored in the cache."""
        if doc_id and doc_id in self.cache and "metadata" in self.cache[doc_id]:
            return self.cache[doc_id]["metadata"]
        return {}

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        if doc_id in self.cache and "text" in self.cache[doc_id]:
            return self.cache[doc_id]["text"]
        return ""

    def split_doc(self, doc, query, data, filter_most_relevant=False):

        def vectorize(text):
            vectorizer = CountVectorizer()
            vectorizer.fit([text])
            return vectorizer

        def main_tokens(tokens):
            return list(
                filter(lambda tok: tok not in self.stop_words and len(tok) > 1,
                       map(lambda tok: tok.lower(), tokens)
                       )
            )

        if doc and data and "window" in data:

            # removes page numbers
            doc = re.sub(r'[\n\r]+[ \t]*\d+[ \t]*[\n\r]+', '\n', doc)
            # removes hyphens
            doc = re.sub(r'-[\r]*\n', '', doc)
            # removes new lines
            doc = re.sub(r'[ \r\n]+', ' ', doc)
            # EM dash
            doc = doc.replace("\u2013", "-").replace("\u2014", "-")
            # US double quotation
            doc = doc.replace("\u201c", "").replace("\u201d", "")
            # US fi character
            doc = doc.replace("\ufb01", "fi")
            # Ellipsis
            doc = doc.replace("\u2026", "...")
            # Vertical tab
            doc = doc.replace("\x0b", ". ")

            sections = []
            current = {"text": [], "tokens": []}

            for sentence in nltk.sent_tokenize(doc, "english"):

                tokens = nltk.word_tokenize(sentence)

                current["tokens"].extend(tokens)
                current["text"].append(sentence)

                if len(current["tokens"]) >= data["window"]:
                    sections.append(
                        {
                            "tokens": " ".join(main_tokens(current["tokens"])),
                            "text": " ".join(current["text"])
                        }
                    )
                    current = {"text": [], "tokens": []}
            if current["tokens"]:
                sections.append(
                    {
                        "tokens": " ".join(main_tokens(current["tokens"])),
                        "text": " ".join(current["text"])
                    }
                )

            if filter_most_relevant or self.filter_most_relevant:
                matrix = []

                try:
                    for section in sections:
                        try:
                            v = vectorize(section["tokens"])
                            result_vector = v.transform([query])[0]
                            matrix.append(sum(result_vector.toarray()[0]))
                        except ValueError:
                            matrix.append(0)
                    max_occurrences = max(matrix)
                except ValueError:
                    max_occurrences = 0

                result = [entry for occ, entry in zip(matrix, sections) if occ == max_occurrences]
                sections = result

            for section in sections:
                yield section["text"]

        else:

            yield doc
