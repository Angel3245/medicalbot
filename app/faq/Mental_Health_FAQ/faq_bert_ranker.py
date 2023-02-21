import sys
sys.path.insert(0, "../../BERT-FAQ")

from elasticsearch import TransportError
from elasticsearch_dsl.connections import connections
from faq_bert_ranker import FAQ_BERT_Ranker

def ranker(top_k, dataset, fields, index_name, loss_type, neg_type, query_type, question):
    try:
        es = connections.create_connection(hosts=['localhost'])
    except TransportError as e:
        e.info()

    version = '1.1'

    model_name = "{}_{}_{}_{}".format(loss_type, neg_type, query_type, version)
    bert_model_path = bert_model_path = "output" + "/" + dataset + "/models/" + model_name

    faq_bert_ranker = FAQ_BERT_Ranker(
        es=es, index=index_name, fields=fields, top_k=top_k, bert_model_path=bert_model_path
    )

    ranked_results = faq_bert_ranker.rank_results(question)
    ranked_result = ranked_results[0]
    print(ranked_result)