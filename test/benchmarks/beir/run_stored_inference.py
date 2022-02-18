import json
import os
import re
import argparse
import pandas as pd
from collections import defaultdict
from haystack.nodes import BaseComponent, JoinDocuments
from haystack.schema import Document
from haystack.pipelines import Pipeline
from run_beir_haystack_evaluation import load_beir_dataset, qrels_to_haystack
from run_beir_retrieval import result_to_qrel_and_dicts

from beir.retrieval.evaluation import EvaluateRetrieval
from datetime import datetime

def load_stored_result(result_path):
    results = {}

    with open(result_path, 'r') as f:
        for line in f:
            query_result = json.loads(line.strip())
            documents = [Document.from_dict(doc) for doc in query_result['pos']]
            query_id = query_result['query_id']
            results[query_id] = {'documents': documents}

    return results


def load_all_stored_results(retriever_names, result_path):
    files = os.listdir(result_path)
    matcher = re.compile('hstack-' + '|hstack-'.join(retriever_names))
    selected_results = [filename for filename in files if re.search(matcher, filename)]
    selected_results = sorted(selected_results)
    retriever_names = sorted(retriever_names)
    query_results = defaultdict(dict)

    for filename, retriever_name in zip(selected_results, retriever_names):
        results = load_stored_result(os.path.join(result_path, filename))
        for query_id, result_dict in results.items():
            query_results[(query_id, retriever_name)] = result_dict

    return query_results

class StoredResultLoader(BaseComponent):
    outgoing_edges = 1

    def __init__(self, retriever_name, stored_results):
        self.set_config(retriever_name=retriever_name, stored_results=stored_results)

        self.stored_results = stored_results
        self.retriever_name = retriever_name

    def run(self, query, labels, top_k=25):
        query_id = labels.labels[0].id
        results = self.stored_results[(query_id, self.retriever_name)]
        results = {'documents': results['documents'][:top_k]}
        return results, 'output_1'


def init_result_loader_nodes(retriever_names, stored_results, pipeline):
    for retriever_name in retriever_names:
        node = StoredResultLoader(retriever_name=retriever_name, stored_results=stored_results)
        pipeline.add_node(node, name=retriever_name, inputs=['Query'])

    return pipeline


def run_stored_inference(
        result_path,
        dataset_name,
        run_name,
        report_path,
        join_top_k,
        top_k,
        retriever_names,
):
    data_path = os.path.join(result_path, dataset_name, 'data')
    result_path = os.path.join(result_path, dataset_name, run_name)
    stored_results = load_all_stored_results(retriever_names=retriever_names, result_path=result_path)
    pipeline = Pipeline()
    pipeline = init_result_loader_nodes(retriever_names=retriever_names, pipeline=pipeline, stored_results=stored_results)
    retriever_params = {retriever_name: {'top_k': top_k} for retriever_name in retriever_names}
    joiner = JoinDocuments(join_mode='reciprocal_rank_fusion')
    pipeline.add_node(joiner, name='Joiner', inputs=retriever_names)

    corpus, queries, qrels = load_beir_dataset(dataset_name, data_path)
    _, labels = qrels_to_haystack(corpus, queries, qrels)

    ################################
    ##### Haystack Evaluation ######
    ################################
    result = pipeline.eval(
        labels=labels,
        params={
            'Joiner': {'top_k_join': join_top_k},
            **retriever_params
        }
    )
    metrics = result.calculate_metrics()
    print('Haystack metrics')
    print(metrics)

    ##################################
    ##### BEIR Evaluation ############
    ##################################
    qrel_results = {}
    qrel_retriever_results = {retriever_name: {} for retriever_name in retriever_names}
    for label in labels:
        results = pipeline.run(
            query=label.query,
            labels=label,
            params={
                'Joiner': {'top_k_join': join_top_k},
                **retriever_params
            }
        )

        qrel, _ = result_to_qrel_and_dicts(label.labels[0].id, results['documents'])
        qrel_results[qrel['query_id']] = qrel['pos']

        for retriever_name in retriever_names:
            query_id = label.labels[0].id
            qrel, _ = result_to_qrel_and_dicts(query_id, stored_results[(query_id, retriever_name)]['documents'])
            qrel_retriever_results[retriever_name][query_id] = qrel['pos']

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, qrel_results, [top_k])
    print('BEIR metrics')
    print('ndcg', ndcg)

    single_retriever_results = {}
    for retriever_name in retriever_names:
        print(retriever_name)
        ndcg_single, _map, _recall, _precision = EvaluateRetrieval.evaluate(qrels, qrel_retriever_results[retriever_name], [top_k])
        single_retriever_results[f'ndcg@{retriever_name}'] = ndcg_single[f'NDCG@{top_k}']


    ##################################
    ##### Report Creation ############
    ##################################
    eval_report_id = f'eval-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print(single_retriever_results)
    with open(os.path.join(report_path, 'evaluations.jsonl'), 'a') as f:
        eval = {
            'eval_id': eval_report_id,
            'dataset': dataset_name,
            'retrievers': ','.join(retriever_names),
            'join_top_k': join_top_k,
            'top_k': top_k,
            'beir_ndcq': ndcg[f'NDCG@{top_k}'],
            'ndcg': metrics['Joiner']['ndcg'],
            'beir_precision': precision[f'P@{top_k}'],
            'precision': metrics['Joiner']['precision'],
            'beir_recall': recall[f'Recall@{top_k}'],
            'recall_single_hit': metrics['Joiner']['recall_single_hit'],
            'recall_multi_hit': metrics['Joiner']['recall_single_hit'],
            **single_retriever_results
        }
        f.write(json.dumps(eval) + '\n')

    ## markdown results
    previous_results = []
    with open(os.path.join(report_path, 'evaluations.jsonl'), 'r') as f:
        for line in f:
            previous_results.append(json.loads(line.strip()))

    result_df = pd.DataFrame(previous_results)
    result_md = result_df.to_markdown()
    with open(os.path.join(report_path, 'evaluations.md'), 'w') as f:
        f.write(result_md)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating BEIR Evaluations')

    parser.add_argument('--retrievers', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--report_path', type=str)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--join_top_k', type=int)
    args = parser.parse_args()

    retrievers = args.retrievers.split(',')

    run_stored_inference(
        result_path=args.result_path,
        top_k=args.top_k,
        join_top_k=args.join_top_k,
        report_path=args.report_path,
        retriever_names=retrievers,
        dataset_name=args.dataset,
        run_name=args.run_name
    )