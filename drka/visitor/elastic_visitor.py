from drka.retriever.elastic_doc_ranker import ElasticDocRanker, ElasticVisitor

class ElasticSearchVisitor(ElasticVisitor):

    def process(self, ranker: ElasticDocRanker, query: str, k: int):

        _body = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "type": "best_fields",
                    "tie_breaker": 0.3,
                    "fields": ranker.elastic_fields_weights,
                    "slop": 100  # ,
                    # "fuzziness": "AUTO"
                }
            },
            "highlight": {
                "fields": {
                    ranker.elastic_field_content: {}
                },
                # TODO: Move these tags to the frontend or use a visitor
                "pre_tags": "<b>",
                "post_tags": "</b>"
            }
        }
        return _body
