import argparse
from pathlib import Path
import pandas as pd
from DB.connect import database_connect
from model.knowledge import Knowledge
from sentence_transformers import SentenceTransformer
from question_similarity import *
from faq.Mental_Health_FAQ import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="select an option", required=True)
    parser.add_argument("-m", "--model", type=str, help="select a model", default='distilbert-base-uncased')
    parser.add_argument("-d", "--dataset", type=str, help="select a dataset", default="MentalFAQ")
    parser.add_argument("-f", "--file", type=str, help="select a file")
    args = parser.parse_args()

    path = Path.cwd()

    if args.option == "insert_csv":
        # python app\main.py -o insert_csv -f <<csv_name>>
        route_csv = F"{str(path)}/file/datasets"+args.file+".csv"

        df = pd.read_csv(route_csv)

        # remove unnecesary columns
        df = df.drop('Question_ID', axis=1)

        # insert data
        session = database_connect("tfgdb")
        session.add(df.values.tolist())

    # Test the semantic search of a SentenceTransformer model with corpus sentences
    if args.option == "test_similarity":
        # python app\main.py -o test_similarity -m <<model_name>> -f <<model_path>>
        # Example: app\main.py -o test_similarity -m all-MiniLM-L6-v2
        session = database_connect("tfgdb")

        qas = session.query(Knowledge).all()
        
        corpus_sentences = []
        for pair in qas:
            corpus_sentences.append([pair.question,pair.answer])

        if (args.file != None):
            model_path = F"{str(path)}"+args.file
            model = SentenceTransformer(model_name_or_path=model_path) # load from file
        else:
            model = SentenceTransformer(args.model) # load pretrained model

        semantic_search(corpus_sentences,model)


    ### BERT-FAQ ###

    # Build and evaluate a complete transformer-based FAQ retrieval system
    if args.option == "createFAQSystem":
        # python app\main.py -o createFAQSystem -d <<dataset_name>> -m <<model_name>>
        # Example: python app\main.py -o createFAQSystem -d MentalFAQ -m distilbert-base-uncased
        model_name = args.model

        if(args.dataset == "MentalFAQ"):
            file_path = F"{str(path)}/file/datasets/Mental_Health_FAQ.csv"
            data_path = F"{str(path)}/file/data/MentalFAQ"
            output_path = F"{str(path)}/output/MentalFAQ"

            parse_mentalfaq(file_path, data_path)
            ingest_data_to_db(data_path)
            generate_hard_negatives(data_path)
            generate_ground_truth_dataset(data_path)
            generate_topk_results(data_path)
            model_training(data_path,output_path,model_name)
            generate_prediction_results(data_path,output_path)
            generate_reranked_results(data_path)
            evaluate(data_path, output_path)
        else:
            print("Dataset not selected")

    #Step 1. - Parsing
    # Parse a FAQ dataset
    if args.option == "parsing_qa":
        # python app\main.py -o parsing_qa -d <<dataset_name>>
        if(args.dataset == "MentalFAQ"):
            file_path = F"{str(path)}/file/datasets/Mental_Health_FAQ.csv"
            data_path = F"{str(path)}/file/data/MentalFAQ"

            parse_mentalfaq(file_path, data_path)
        else:
            print("Dataset not selected")

    #Step 2. - Ingest Data to DB
    # Insert previously parsed data into a DB. It needs a connection to a DB (ElasticSearch)
    if args.option == "ingest_db":
        # python app\main.py -o ingest_db -d <<dataset_name>>
        if(args.dataset == "MentalFAQ"):
            data_path = F"{str(path)}/file/data/MentalFAQ"

            ingest_data_to_db(data_path)
        else:
            print("Dataset not selected")

    #Step 3. - Generate Hard Negatives
    # Generate hard negative samples
    if args.option == "generate_hard":
        # python app\main.py -o generate_hard -d <<dataset_name>>
        if(args.dataset == "MentalFAQ"):
            data_path = F"{str(path)}/file/data/MentalFAQ"

            generate_hard_negatives(data_path)
        else:
            print("Dataset not selected")

    #Step 4. - Generate Ground Truth Dataset
    # Generate triplet dataset for BERT finetuning
    if args.option == "generate_gt":
        # python app\main.py -o generate_gt -d <<dataset_name>>
        if(args.dataset == "MentalFAQ"):
            data_path = F"{str(path)}/file/data/MentalFAQ"

            generate_ground_truth_dataset(data_path)
        else:
            print("Dataset not selected")

    #Step 5. - Generate ES Topk Results For Reranking
    # Generate ES top-k results for re-ranking
    if args.option == "generate_topk":
        # python app\main.py -o generate_topk -d <<dataset_name>>
        if(args.dataset == "MentalFAQ"):
            data_path = F"{str(path)}/file/data/MentalFAQ"
            generate_topk_results(data_path)
        else:
            print("Dataset not selected")

    #Step 6. - Model training
    # Finetune BERT model
    if args.option == "model_training":
        # python app\main.py -o model_training -d <<dataset_name>> -m <<model_name>>
        if(args.dataset == "MentalFAQ"):
            data_path = F"{str(path)}/file/data/MentalFAQ"
            output_path = F"{str(path)}/output/MentalFAQ"
            model_name = args.model

            model_training(data_path,output_path,model_name)
        else:
            print("Dataset not selected")

    #Step 7. - Generate BERT Prediction Results
    # Generate BERT prediction results
    if args.option == "generate_results":
        # python app\main.py -o generate_results -d <<dataset_name>>
        if(args.dataset == "MentalFAQ"):
            data_path = F"{str(path)}/file/data/MentalFAQ"
            model_path = F"{str(path)}/output/MentalFAQ"

            generate_prediction_results(data_path,model_path)
        else:
            print("Dataset not selected")

    #Step 8. - Generate Reranked Results
    # Perform re-ranking
    if args.option == "generate_reranked_results":
        # python app\main.py -o generate_reranked_results -d <<dataset_name>>
        if(args.dataset == "MentalFAQ"):
            data_path = F"{str(path)}/file/data/MentalFAQ"
            generate_reranked_results(data_path)
        else:
            print("Dataset not selected")

    #Step 9. - Evaluation
    # Generate evaluation metrics: NDCG@3, NDCG@5, NDCG@10, P@3, P@5, P@10, MAP
    if args.option == "evaluation":
        # python app\main.py -o evaluation -d <<dataset_name>>
        if(args.dataset == "MentalFAQ"):
            data_path = F"{str(path)}/file/data/MentalFAQ"
            output_path = F"{str(path)}/output/MentalFAQ"

            evaluate(data_path, output_path)
        else:
            print("Dataset not selected")

    ###
    print("PROGRAM FINISHED")