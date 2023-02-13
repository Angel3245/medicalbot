import sys
sys.path.insert(0, '../../BERT-FAQ/')

# import required libraries
from elasticsearch_dsl import Index, Document, Integer, Text, analyzer, Keyword, Double
from elasticsearch_dsl.connections import connections
from elasticsearch import Elasticsearch, helpers

from evaluation import get_relevance_label_df
from shared.utils import load_from_json
from shared.utils import dump_to_json
from shared.utils import make_dirs
from indexer import ingest_data
from indexer import QA
import logging

def ingest_data_to_db(data_path):
    # get list of query answer pairs
    query_answer_pairs_filepath = data_path+'/query_answer_pairs.json'
    relevance_label_df = get_relevance_label_df(query_answer_pairs_filepath)
    faq_qa_pair_df = relevance_label_df[relevance_label_df['query_type'] == 'faq']
    faq_qa_pairs = faq_qa_pair_df.T.to_dict().values()

    # convert to list
    faq_qa_pairs = list(faq_qa_pairs)

    try:
        
        es = connections.create_connection(hosts=['localhost'])
        
        # Index data to Elasticsearch 
        index_name = "mentalfaq"
        
        # Initialize index (only perform once)
        index = Index(index_name)

        # Define custom settings
        index.settings(
            number_of_shards=1,
            number_of_replicas=0
        )

        # Delete the index, ignore if it doesn't exist
        index.delete(ignore=404)

        # Create the index in Elasticsearch
        index.create()

        # Register a document with the index
        index.document(QA)

        # Ingest data to Elasticsearch
        ingest_data(faq_qa_pairs, es=es, index=index_name)

        print("Finished indexing {} records to {} index".format(len(faq_qa_pairs), index_name))

    except Exception:
        logging.error('exception occured', exc_info=True)