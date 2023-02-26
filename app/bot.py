import argparse
from pathlib import Path
from faq.Mental_Health_FAQ import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='distilbert-base-uncased')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "retrieve_answer":
        # python app\bot.py -o retrieve_answer
        top_k = 100
        dataset = 'MentalFAQ'
        fields = ['question_answer']
        index_name = "mentalfaq"

        # Define model parameters
        loss_type = 'triplet'; neg_type = 'hard'; query_type = 'user_query'

        # Question
        question = "What is a mental illness?"

        ranker(top_k, dataset, fields, index_name, loss_type, neg_type, query_type, question)


    print("PROGRAM FINISHED")