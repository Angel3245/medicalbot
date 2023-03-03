from elasticsearch import Elasticsearch

import sys
sys.path.insert(0, '../../BERT-FAQ/')

from evaluation import get_relevance_label_df
from evaluation import get_relevance_label
from hard_negatives_generator import Hard_Negatives_Generator

# import utility functions
from shared.utils import load_from_json
from shared.utils import dump_to_json
from shared.utils import make_dirs

def generate_hard_negatives(data_path, index_name):
    query_answer_pair_filepath = data_path+'/query_answer_pairs.json'
    relevance_label_df = get_relevance_label_df(query_answer_pair_filepath) 

    es = Elasticsearch([{'host':'localhost','port':9200}], http_auth=('username', 'password')) 

    hng = Hard_Negatives_Generator(
        es=es, index=index_name, query_by=['question_answer'], top_k=100, query_type='faq')
    hard_negatives_faq = hng.get_hard_negatives(relevance_label_df)
    #print(hard_negatives_faq)
    dump_to_json(hard_negatives_faq, data_path+'/hard_negatives_faq.json')

    hng = Hard_Negatives_Generator(
        es=es, index=index_name, query_by=['question_answer'], top_k=100, query_type='user_query')
    hard_negatives_user_query = hng.get_hard_negatives(relevance_label_df)

    #print(hard_negatives_user_query)
    dump_to_json(hard_negatives_user_query, data_path+'/hard_negatives_user_query.json')