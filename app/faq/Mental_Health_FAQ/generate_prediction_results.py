import sys
sys.path.insert(0, "../../BERT-FAQ")

from evaluation import get_relevance_label_df
from shared.utils import load_from_json
from shared.utils import dump_to_json
from shared.utils import make_dirs
from reranker import ReRanker

def generate_prediction_results(data_path,model_path):
    output_path=data_path+"/rank_results"

    # load user_query ES results from json files
    es_output_path = output_path + "/unsupervised"
    es_query_by_question = load_from_json(es_output_path + '/es_query_by_question.json')
    es_query_by_answer = load_from_json(es_output_path + '/es_query_by_answer.json')
    es_query_by_question_answer = load_from_json(es_output_path + '/es_query_by_question_answer.json')
    es_query_by_question_answer_concat = load_from_json(es_output_path + '/es_query_by_question_answer_concat.json')

    # load test_queries, relevance_label_df for ReRanker
    query_answer_pair_filepath = data_path+'/query_answer_pairs.json'
    relevance_label_df = get_relevance_label_df(query_answer_pair_filepath)
    test_queries = relevance_label_df[relevance_label_df['query_type'] == 'user_query'].question.unique()

    version="1.1"
    # define rank_field parameter
    for rank_field in ["BERT-Q-a","BERT-Q-q"]: 

        # define variables
        for query_type in ["user_query","faq"]:
            for neg_type in ["hard","simple"]:
                for loss_type in ["triplet","softmax"]:
                    bert_model_path=model_path+'/models/' + loss_type + '_' + neg_type + '_' + query_type + '_' + version

                    # create instance of ReRanker class
                    r = ReRanker(
                        bert_model_path=bert_model_path, 
                        test_queries=test_queries, relevance_label_df=relevance_label_df,
                        rank_field=rank_field
                    )

                    # generate directory structure
                    pred_output_path = output_path + "/supervised/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type
                    make_dirs(pred_output_path)

                    # next, generate BERT, Re-ranked top-k results and dump to files
                    bert_query_by_question = r.get_bert_topk_preds(es_query_by_question)
                    dump_to_json(bert_query_by_question, pred_output_path + '/bert_query_by_question.json')

                    bert_query_by_answer = r.get_bert_topk_preds(es_query_by_answer)
                    dump_to_json(bert_query_by_answer, pred_output_path + '/bert_query_by_answer.json')

                    bert_query_by_question_answer = r.get_bert_topk_preds(es_query_by_question_answer)
                    dump_to_json(bert_query_by_question_answer, pred_output_path + '/bert_query_by_question_answer.json')

                    bert_query_by_question_answer_concat = r.get_bert_topk_preds(es_query_by_question_answer_concat)
                    dump_to_json(bert_query_by_question_answer_concat, pred_output_path + '/bert_query_by_question_answer_concat.json')