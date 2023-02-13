import sys
sys.path.insert(0, "../../BERT-FAQ")

from shared.utils import dump_to_json
from parsing.mentalfaq import MentalFAQ_Parser
import pandas as pd
from shared.utils import make_dirs

def parse_mentalfaq(file_path, output_path):
    # read data as pandas DataFrame
    df = pd.read_csv(file_path)

    # rename colnames question1: query_string, question2: question
    df = df.rename(columns={'question1': 'query_string', 'Questions': 'question','Answers':'answer'})

    # drop null values
    df.dropna(inplace=True)

    # create instance of MentalFAQ_Parser and generate query_answer_pairs
    mentalfaq_parser = MentalFAQ_Parser()
    mentalfaq_parser.extract_data(df)

    # get query_answer_pairs
    query_answer_pairs = mentalfaq_parser.query_answer_pairs

    # Dump data to json file
    make_dirs(output_path)
    dump_to_json(query_answer_pairs, output_path+'/query_answer_pairs.json', sort_keys=False)