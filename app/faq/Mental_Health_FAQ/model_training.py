import sys
import os
import torch
sys.path.insert(0, "../../BERT-FAQ")

# import required libraries
import pandas as pd
from faq_bert_finetuning import FAQ_BERT_Finetuning
from shared.utils import make_dirs

def model_training(data_path,output_path,model_name):
    # define output path to save model & evaluation files
    make_dirs(output_path)

    # load hard_faq_dataset.csv
    hard_faq_dataset = pd.read_csv(data_path+'/dataset/triplet/faq/hard_faq_dataset.csv')

    # create instance of FAQ_BERT_Finetuning
    bert_hard_faq = FAQ_BERT_Finetuning(
        loss_type="triplet", query_type='faq', neg_type='hard', version="1.1", epochs=4, batch_size=16, 
        pre_trained_name=model_name, 
        evaluation_steps=1000, test_size=0.2, learning_rate=5e-5, adam_lrs=5e-5
    )

    
    # create model and save to output path
    bert_hard_faq.create_model(hard_faq_dataset, output_path)

    return
    # load simple_faq_dataset.csv
    simple_faq_dataset = pd.read_csv(data_path+'/dataset/triplet/faq/simple_faq_dataset.csv')

    # create instance of FAQ_BERT_Finetuning
    bert_simple_faq = FAQ_BERT_Finetuning(
        loss_type="triplet", query_type='faq', neg_type='simple', version="1.1", epochs=4, batch_size=16, 
        pre_trained_name='distilbert-base-uncased', 
        evaluation_steps=1000, test_size=0.2
    )

    # create model and save to output path
    bert_simple_faq.create_model(simple_faq_dataset, output_path)

    # load hard_user_query_dataset.csv
    hard_user_query_dataset = pd.read_csv(data_path+'/dataset/triplet/user_query/hard_user_query_dataset.csv')

    # create instance of FAQ_BERT_Finetuning
    bert_hard_user_query = FAQ_BERT_Finetuning(
        loss_type="triplet", query_type='user_query', neg_type='hard', version="1.1", epochs=4, batch_size=16, 
        pre_trained_name='distilbert-base-uncased', 
        evaluation_steps=1000, test_size=0.2
    )

    # create model and save to output path
    bert_hard_user_query.create_model(hard_user_query_dataset, output_path)

    # load simple_user_query_dataset.csv
    simple_user_query_dataset = pd.read_csv(data_path+'/dataset/triplet/user_query/simple_user_query_dataset.csv')

    # create instance of FAQ_BERT_Finetuning
    bert_simple_user_query = FAQ_BERT_Finetuning(
        loss_type="triplet", query_type='user_query', neg_type='simple', version="1.1", epochs=4, batch_size=16, 
        pre_trained_name='distilbert-base-uncased', 
        evaluation_steps=1000, test_size=0.2
    )

    # create model and save to output path
    bert_simple_user_query.create_model(simple_user_query_dataset, output_path)

    # load hard_faq_dataset.csv
    hard_faq_dataset = pd.read_csv(data_path+'/dataset/softmax/faq/hard_faq_dataset.csv')

    # create instance of FAQ_BERT_Finetuning
    bert_hard_faq = FAQ_BERT_Finetuning(
        loss_type="softmax", query_type='faq', neg_type='hard', version="1.1", epochs=4, batch_size=16, 
        pre_trained_name='distilbert-base-uncased', 
        evaluation_steps=1000, test_size=0.2
    )

    # create model and save to output path
    bert_hard_faq.create_model(hard_faq_dataset, output_path)

    # load simple_faq_dataset.csv
    simple_faq_dataset = pd.read_csv(data_path+'/dataset/softmax/faq/simple_faq_dataset.csv')

    # create instance of FAQ_BERT_Finetuning
    bert_simple_faq = FAQ_BERT_Finetuning(
        loss_type="softmax", query_type='faq', neg_type='simple', version="1.1", epochs=4, batch_size=16, 
        pre_trained_name='distilbert-base-uncased', 
        evaluation_steps=1000, test_size=0.2
    )

    # create model and save to output path
    bert_simple_faq.create_model(simple_faq_dataset, output_path)

    # load hard_user_query_dataset.csv
    hard_user_query_dataset = pd.read_csv(data_path+'/dataset/softmax/user_query/hard_user_query_dataset.csv')

    # create instance of FAQ_BERT_Finetuning
    bert_hard_user_query = FAQ_BERT_Finetuning(
        loss_type="softmax", query_type='user_query', neg_type='hard', version="1.1", epochs=4, batch_size=16, 
        pre_trained_name='distilbert-base-uncased', 
        evaluation_steps=1000, test_size=0.2
    )

    # create model and save to output path
    bert_hard_user_query.create_model(hard_user_query_dataset, output_path)

    # load simple_user_query_dataset.csv
    simple_user_query_dataset = pd.read_csv(data_path+'/dataset/softmax/user_query/simple_user_query_dataset.csv')

    # create instance of FAQ_BERT_Finetuning
    bert_simple_user_query = FAQ_BERT_Finetuning(
        loss_type="softmax", query_type='user_query', neg_type='simple', version="1.1", epochs=4, batch_size=16, 
        pre_trained_name='distilbert-base-uncased', 
        evaluation_steps=1000, test_size=0.2
    )

    # create model and save to output path
    bert_simple_user_query.create_model(simple_user_query_dataset, output_path)