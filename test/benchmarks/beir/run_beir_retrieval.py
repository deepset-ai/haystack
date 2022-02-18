import argparse
import os
import shutil
import json
import time
from pathlib import Path
from datetime import datetime
from run_beir_haystack_evaluation import load_beir_dataset, qrels_to_haystack, index_documents, init_doc_store
from haystack.utils import launch_es


def result_to_qrel_and_dicts(query_id, result):
    qrels = {'query_id': query_id, 'pos': {doc.id: doc.score for doc in result}}
    result_dict = {'query_id': query_id, 'pos': [doc.to_dict() for doc in result]}

    return qrels, result_dict


def convert_and_save_result(query_id, result, filename, save_path):
    qrels, result_dict = result_to_qrel_and_dicts(query_id, result)
    qrel_path = os.path.join(save_path, f'qrel-{filename}.jsonl')
    result_dict_path = os.path.join(save_path, f'hstack-{filename}.jsonl')

    with open(qrel_path, 'a') as f:
        f.write(json.dumps(qrels) + '\n')

    with open(result_dict_path, 'a') as f:
        f.write(json.dumps(result_dict) + '\n')


def run_single_retriever(
        retriever_name,
        documents,
        labels,
        result_output_path,
        logfile,
        top_k=25,
        max_seq_len=300,
        emb_dim=768,
):
    start_time = time.process_time()
    print(f'Starting to run {retriever_name} for {len(documents)} documents and {len(labels)} queries.')
    document_store = init_doc_store(launch=False, embedding_dim=emb_dim, retriever_name=retriever_name)
    retriever = index_documents(documents=documents, document_store=document_store, retriever_name=retriever_name, max_seq_len=max_seq_len)
    queries = [multi_label.query for multi_label in labels]
    query_ids = [multi_label.labels[0].id for multi_label in labels]
    result_filename_part = f'{retriever_name.split("/")[-1]}-topk-{top_k}-seq-len-{max_seq_len}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    if retriever_name in ['bm25', 'doc2query']:
        for idx, query in enumerate(queries):
            result = retriever.retrieve(query=query, top_k=top_k)
            query_id = query_ids[idx]
            convert_and_save_result(query_id=query_id, result=result, filename=result_filename_part, save_path=result_output_path)
    else:
        query_embeddings = retriever.embed_queries(queries)
        for idx, query_embedding in enumerate(query_embeddings):
            result = document_store.query_by_embedding(query_emb=query_embedding, top_k=top_k)
            query_id = query_ids[idx]
            convert_and_save_result(query_id=query_id, result=result, filename=result_filename_part,
                                    save_path=result_output_path)


    stop_time = time.process_time() - start_time
    print(f'Finished to retrieve {len(documents)} documents and {len(labels)} queries in {stop_time} seconds.')
    with open(logfile, 'a') as f:
        f.write(json.dumps({'retriever': retriever_name.split('/')[-1], 'duration': stop_time, 'start': start_time}) + '\n')


def run_single_dataset_retrieval(
        retrievers,
        result_base_path,
        run_name,
        dataset_name,
        top_k,
        logfile
):
    result_output_path = Path(result_base_path) / f'{dataset_name}/{run_name}'
    if result_output_path.exists() and result_output_path.is_dir():
        shutil.rmtree(result_output_path)

    dataset_save_path = Path(result_base_path) / f'{dataset_name}/data'
    dataset_save_path.mkdir(parents=True, exist_ok=True)
    result_output_path.mkdir(parents=True, exist_ok=True)

    start_time = time.process_time()
    print(f'Starting to run {len(retrievers)} retrievers for {dataset_name} dataset.')
    corpus, queries, qrels = load_beir_dataset(dataset_name=dataset_name, save_path=dataset_save_path)
    documents, labels = qrels_to_haystack(corpus, queries, qrels)

    for retriever_name, embedding_dim, max_seq_len in retrievers:
        run_single_retriever(
            retriever_name=retriever_name,
            documents=documents[:2],
            labels=labels[:2],
            top_k=top_k,
            max_seq_len=max_seq_len,
            emb_dim=embedding_dim,
            result_output_path=result_output_path,
            logfile=logfile
        )
    stop_time = time.process_time() - start_time
    print(f'Finished running {len(retrievers)} retrievers for {dataset_name} dataset in {stop_time} seconds.')
    with open(logfile, 'a') as f:
        f.write(json.dumps({'dataset': dataset_name, 'duration': stop_time, 'start': start_time}) + '\n')


def run_dataset_retrievals(
        datasets,
        retriever_names,
        emb_dims,
        max_seq_lens,
        top_k,
        logfile,
        result_base_path,
        run_name
):
    retrievers = list(zip(retriever_names, emb_dims, max_seq_lens))

    for dataset in datasets:
        run_single_dataset_retrieval(
            retrievers=retrievers,
            result_base_path=result_base_path,
            run_name=run_name,
            top_k=top_k,
            logfile=logfile,
            dataset_name=dataset
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating BEIR Inference')

    parser.add_argument('--datasets', type=str)
    parser.add_argument('--retrievers', type=str)
    parser.add_argument('--emb_dims', type=str)
    parser.add_argument('--max_lens', type=str)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--run', type=str)
    args = parser.parse_args()
    launch_es(30)
    datasets = args.datasets.split(',')
    retrievers = args.retrievers.split(',')
    emb_dims = [int(dim) for dim in args.emb_dims.split(',')]
    max_seq_lens = [int(length) for length in args.max_lens.split(',')]
    top_k = args.top_k

    output_path = os.path.join(os.getcwd(), '../reports')
    logfile = os.path.join(output_path, 'logfile.jsonl')

    print(args)
    run_dataset_retrievals(
        datasets=datasets,
        retriever_names=retrievers,
        emb_dims=emb_dims,
        max_seq_lens=max_seq_lens,
        logfile=logfile,
        top_k=top_k,
        result_base_path=output_path,
        run_name=args.run
    )