from shared.utils import dump_to_json
from parsers.reddit import Reddit_Parser
import pandas as pd
from shared.utils import make_dirs

def parse_redditposts_questions(posts_path, comments_path, output_path):
    # read data as pandas DataFrame
    posts_df = pd.read_csv(posts_path)
    comments_df = pd.read_csv(comments_path)

    # drop null values
    posts_df.dropna(inplace=True)
    comments_df.dropna(inplace=True)

    #filter posts columns
    columns = posts_df.columns
    columns_to_keep = ["title", "body", "score", "name", "link_flair_text"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    posts_df = posts_df.drop(columns_to_remove, axis=1)
    # rename colnames body: post_body, score: post_score
    posts_df = posts_df.rename(columns={'body': 'post_body', 'score': 'post_score'})

    # filter comments columns
    columns = comments_df.columns
    columns_to_keep = ["body", "score", "parent_id"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    comments_df = comments_df.drop(columns_to_remove, axis=1)

    # filter by flair
    flairs = ["DAE Questions", "Question", ":snoo_thoughtful: help? :snoo_biblethump:",
              ":orly: Help please!", "DAE?"]
    posts_df = posts_df.apply(lambda row: row[posts_df['link_flair_text'].isin(flairs)])

    # create a new comments_length column that contains the number of words per comment:
    comments_df["comment_length"] = comments_df.apply(
        lambda x: len(x["body"].split()), axis=1
    )

    # filter out short comments, which typically include things “Thanks!” that are not relevant for our search engine.
    comments_df = comments_df[comments_df["comment_length"] > 15]

    # remove answers with low scores
    comments_df = comments_df[comments_df['score'] > 3]

    # get answer with best score
    comments_df = comments_df[comments_df.groupby(['parent_id'], group_keys=False)['score'].transform(max) == comments_df['score']]

    # join both dataframes
    df = posts_df.merge(comments_df,left_on='name', right_on='parent_id')

    # concatenate the title and post_body together in a new question column
    df['question'] = df['title'].map(str) + ' ' + df['post_body'].map(str)

    # rename colname body: answer
    df = df.rename(columns={'body': 'answer'})
    
    # create instance of Reddit_Parser and generate query_answer_pairs
    mentalfaq_parser = Reddit_Parser()
    mentalfaq_parser.extract_data(df)

    # get query_answer_pairs
    query_answer_pairs = mentalfaq_parser.query_answer_pairs

    # Dump data to json file
    make_dirs(output_path)
    dump_to_json(query_answer_pairs, output_path+'/query_answer_pairs.json', sort_keys=False)

def parse_redditposts_support(posts_path, comments_path, output_path):
    # read data as pandas DataFrame
    posts_df = pd.read_csv(posts_path)
    comments_df = pd.read_csv(comments_path)

    # drop null values
    posts_df.dropna(inplace=True)
    comments_df.dropna(inplace=True)

    #filter posts columns
    columns = posts_df.columns
    columns_to_keep = ["title", "body", "score", "name", "link_flair_text"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    posts_df = posts_df.drop(columns_to_remove, axis=1)
    # rename colnames body: post_body, score: post_score
    posts_df = posts_df.rename(columns={'body': 'post_body', 'score': 'post_score'})

    # filter comments columns
    columns = comments_df.columns
    columns_to_keep = ["body", "score", "parent_id"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    comments_df = comments_df.drop(columns_to_remove, axis=1)

    # filter by flair
    flairs = ["Needs A Hug/Support", "Need Support", ":snoo_hug: send support :snoo_sad:", "Advice", "Advice Needed", "Support", "Seeking Support"]
    posts_df = posts_df.apply(lambda row: row[posts_df['link_flair_text'].isin(flairs)])

    # create a new comments_length column that contains the number of words per comment:
    comments_df["comment_length"] = comments_df.apply(
        lambda x: len(x["body"].split()), axis=1
    )

    # filter out short comments, which typically include things “Thanks!” that are not relevant for our search engine.
    comments_df = comments_df[comments_df["comment_length"] > 15]

    # remove answers with low scores
    comments_df = comments_df[comments_df['score'] > 3]

    # get answer with best score
    comments_df = comments_df[comments_df.groupby(['parent_id'], group_keys=False)['score'].transform(max) == comments_df['score']]

    # join both dataframes
    df = posts_df.merge(comments_df,left_on='name', right_on='parent_id')

    # concatenate the title and post_body together in a new question column
    df['question'] = df['title'].map(str) + ' ' + df['post_body'].map(str)

    # rename colname body: answer
    df = df.rename(columns={'body': 'answer'})
    
    # create instance of MentalFAQ_Parser and generate query_answer_pairs
    mentalfaq_parser = Reddit_Parser()
    mentalfaq_parser.extract_data(df)

    # get query_answer_pairs
    query_answer_pairs = mentalfaq_parser.query_answer_pairs

    # Dump data to json file
    make_dirs(output_path)
    dump_to_json(query_answer_pairs, output_path+'/query_answer_pairs.json', sort_keys=False)