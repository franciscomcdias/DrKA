import json
import os

from flask import Flask, render_template, request, current_app

from drka import pipeline

app = Flask(__name__, static_url_path='/static')

drka_data_directory = '../data'

config = {
    'reader-model': os.path.join(drka_data_directory, '', 'reader.model'),
    'retriever-model': os.path.join(drka_data_directory, '', 'qa-tfidf-ngram=2-hash=16777216-tokenizer=spacy.npz'),
    'doc-db': os.path.join(drka_data_directory, '', 'qa.db'),
    'embedding-file': None,
    'tokenizer': 'spacy',
    'no-cuda': True,
    'gpu': 0
}


def process(question, candidates=None, top_n=1, n_docs=1):
    with app.app_context():
        predictions = current_app.DrKA.process(
            question, candidates, top_n, n_docs, context={"return": True}
        )
        answers = []
        for i, p in enumerate(predictions, 1):
            answers.append({
                'index': i,
                'span': p['span'],
                'doc_id': p['doc_id'],
                'span_score': '%.5g' % p['span_score'],
                'doc_score': '%.5g' % p['doc_score'],
                'text': p['context']['text'],
                'start': p['context']['start'],
                'end': p['context']['end']
            })

    return answers


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    answers = process(question=data['question'])
    return json.dumps(answers)


def initialize():
    DrKA = pipeline.DrKA(
        cuda=False,
        reader_model=config['reader-model'],
        ranker_config={'options': {'tfidf_path': config['retriever-model']}},
        db_config={'options': {'db_path': config['doc-db']}},
        tokenizer=config['tokenizer'],
        embedding_file=config['embedding-file'],
    )
    return DrKA


if __name__ == '__main__':
    with app.app_context():
        current_app.DrKA = initialize()
    app.run()
