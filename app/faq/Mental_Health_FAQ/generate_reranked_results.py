# import required dependencies
import sys
sys.path.insert(0, '../../BERT-FAQ/')

from shared.utils import load_from_json
from shared.utils import dump_to_json
from shared.utils import make_dirs
from reranker import ReRanker

def generate_reranked_results(data_path):
    # define output path
    output_path=data_path+"/rank_results"

    # define rank_field, w_t parameters
    w_t=10
    for rank_field in ["BERT-Q-a","BERT-Q-q"]: 

        # define variables
        for query_type in ["user_query","faq"]:
            for neg_type in ["hard","simple"]:
                for loss_type in ["triplet","softmax"]:
                    # create instance of ReRanker class
                    r = ReRanker(rank_field=rank_field, w_t=w_t)

                    reranked_output_path = output_path + "/supervised/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type
                    pred_output_path     = output_path + "/supervised/" + rank_field + "/" + loss_type + "/" + query_type + "/" + neg_type

                    # generate reranked results 
                    bert_query_by_question = load_from_json(pred_output_path + '/bert_query_by_question.json')
                    reranked_query_by_question = r.get_reranked_results(bert_query_by_question)
                    dump_to_json(reranked_query_by_question, reranked_output_path + '/reranked_query_by_question.json')

                    bert_query_by_answer = load_from_json(pred_output_path + '/bert_query_by_answer.json')
                    reranked_query_by_answer = r.get_reranked_results(bert_query_by_answer)
                    dump_to_json(reranked_query_by_answer, reranked_output_path + '/reranked_query_by_answer.json')

                    bert_query_by_question_answer = load_from_json(pred_output_path + '/bert_query_by_question_answer.json')
                    reranked_query_by_question_answer = r.get_reranked_results(bert_query_by_question_answer)
                    dump_to_json(reranked_query_by_question_answer, reranked_output_path + '/reranked_query_by_question_answer.json')

                    bert_query_by_question_answer_concat = load_from_json(pred_output_path + '/bert_query_by_question_answer_concat.json')
                    reranked_query_by_question_answer_concat = r.get_reranked_results(bert_query_by_question_answer_concat)
                    dump_to_json(reranked_query_by_question_answer_concat, reranked_output_path + '/reranked_query_by_question_answer_concat.json')