import sys
sys.path.insert(0, '../../BERT-FAQ/')

# import class to generate training data
from training_data_generator import Training_Data_Generator

# import utility functions
from shared.utils import load_from_json
from shared.utils import dump_to_json
from shared.utils import make_dirs

def generate_ground_truth_dataset(data_path):

    # read query_answer_pairs
    filepath = data_path+'/query_answer_pairs.json'

    query_answer_pairs = load_from_json(filepath)

    ####################### Triplet Loss ########################
    # simple Type
    tdg = Training_Data_Generator(
        random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
        neg_type='simple', query_type='faq', loss_type='triplet'
    )
    tdg.generate_triplet_dataset(
        query_answer_pairs=query_answer_pairs, 
        output_path=data_path+'/'
    )


    # hard Type
    tdg = Training_Data_Generator(
        random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
        neg_type='hard', query_type='faq', loss_type='triplet'
    )
    tdg.generate_triplet_dataset(
        query_answer_pairs=query_answer_pairs, 
        output_path=data_path+'/'
    )


    tdg = Training_Data_Generator(
        random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
        neg_type='simple', query_type='user_query', loss_type='triplet'
    )
    tdg.generate_triplet_dataset(
        query_answer_pairs=query_answer_pairs, 
        output_path=data_path+'/'
    )


    # hard Type
    tdg = Training_Data_Generator(
        random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
        neg_type='hard', query_type='user_query', loss_type='triplet'
    )
    tdg.generate_triplet_dataset(
        query_answer_pairs=query_answer_pairs, 
        output_path=data_path+'/'
    )


    ####################### Softmax Loss ########################
    # simple Type
    tdg = Training_Data_Generator(
        random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
        neg_type='simple', query_type='faq', loss_type='softmax'
    )
    tdg.generate_triplet_dataset(
        query_answer_pairs=query_answer_pairs, 
        output_path=data_path+'/'
    )

    # hard Type
    tdg = Training_Data_Generator(
        random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
        neg_type='hard', query_type='faq', loss_type='softmax'
    )
    tdg.generate_triplet_dataset(
        query_answer_pairs=query_answer_pairs, 
        output_path=data_path+'/'
    )

    # simple Type
    tdg = Training_Data_Generator(
        random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
        neg_type='simple', query_type='user_query', loss_type='softmax'
    )
    tdg.generate_triplet_dataset(
        query_answer_pairs=query_answer_pairs, 
        output_path=data_path+'/'
    )

    # hard Type
    tdg = Training_Data_Generator(
        random_seed=5, num_samples=24, hard_filepath=data_path+'/', 
        neg_type='hard', query_type='user_query', loss_type='softmax'
    )
    tdg.generate_triplet_dataset(
        query_answer_pairs=query_answer_pairs, 
        output_path=data_path+'/'
    )