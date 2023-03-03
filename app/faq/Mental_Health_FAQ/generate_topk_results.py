from elasticsearch import Elasticsearch

import sys
sys.path.insert(0, '../../BERT-FAQ/')

# import required classes and functions
from reranker import ReRanker
from evaluation import get_relevance_label_df

# import utility functions
from shared.utils import load_from_json
from shared.utils import dump_to_json
from shared.utils import make_dirs

def generate_topk_results(data_path, index_name, top_k=100):
    # Generate list of test queries, relevance labels for ReRanker class
    query_answer_pair_filepath = data_path+'/query_answer_pairs.json'
    relevance_label_df = get_relevance_label_df(query_answer_pair_filepath)
    test_queries = relevance_label_df[relevance_label_df['query_type'] == 'user_query'].question.unique()

    test_queries = list(test_queries)

    # Define instance of ReRanker class
    r = ReRanker(
        bert_model_path='', 
        test_queries=test_queries, relevance_label_df=relevance_label_df
    )

    # create output path to save Elasticsearch top-k results
    output_path = data_path+"/rank_results/unsupervised"
    make_dirs(output_path)


    # Get top-k Elasticsearch results 

    es = Elasticsearch([{'host':'localhost','port':9200}], http_auth=('elastic', 'elastic')) 

    es_query_by_question = r.get_es_topk_results(es=es, index=index_name, query_by=['question'], top_k=top_k)
    es_query_by_answer = r.get_es_topk_results(es=es, index=index_name, query_by=['answer'], top_k=top_k)
    es_query_by_question_answer = r.get_es_topk_results(es=es, index=index_name, query_by=['question', 'answer'], top_k=top_k)
    es_query_by_question_answer_concat = r.get_es_topk_results(es=es, index=index_name, query_by=['question_answer'], top_k=top_k)

    # Save Elasticsearch results to json files
    dump_to_json(es_query_by_question, output_path + '/es_query_by_question.json')
    dump_to_json(es_query_by_answer, output_path + '/es_query_by_answer.json')
    dump_to_json(es_query_by_question_answer, output_path + '/es_query_by_question_answer.json')
    dump_to_json(es_query_by_question_answer_concat, output_path + '/es_query_by_question_answer_concat.json')