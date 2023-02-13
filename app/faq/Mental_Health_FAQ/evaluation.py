import sys
sys.path.insert(0, '../../BERT-FAQ/')

from evaluation import Evaluation

def evaluate(data_path,output_path):
    rank_results_filepath=data_path+"/rank_results"

    ev = Evaluation()
    df = ev.get_eval_df(rank_results_filepath)

    df.to_csv(output_path+"/mentalfaq_evaluation.csv")