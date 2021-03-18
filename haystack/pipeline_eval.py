
from farm.evaluation.squad_evaluation import compute_f1 as calculate_f1_str
from farm.evaluation.squad_evaluation import compute_exact as calculate_em_str

def main():
    document_store_es = ElasticsearchDocumentStore()
    es_retriever = ElasticsearchRetriever(document_store=document_store_es)
    eval_retriever = EvalRetriever()
    reader = FARMReader("deepset/roberta-base-squad2", top_k_per_candidate=4, num_processes=1)
    eval_reader = EvalReader()

    # Here is the pipeline definition
    p = Pipeline()
    p.add_node(component=es_retriever, name="ESRetriever", inputs=["Query"])
    p.add_node(component=eval_retriever, name="EvalRetriever", inputs=["ESRetriever"])
    p.add_node(component=reader, name="QAReader", inputs=["EvalRetriever"])
    p.add_node(component=eval_reader, name="EvalReader", inputs=["QAReader"])

    results = []
    for i, (q, l) in enumerate(q_to_l_dict):
        if i % 10 == 0:
            print(i)
        res = p.run(query=q, top_k_retriever=top_k_retriever, labels=l, top_k_reader=10, skip_incorrect_retrieval=True)
        results.append(res)

def print_eval_metrics(eval_retriever, eval_reader):
    total_queries = eval_retriever.query_count
    retriever_recall = eval_retriever.recall
    correct_retrieval = eval_retriever.correct_retrieval
    reader_top_1_em = eval_reader.top_1_em
    reader_top_k_em = eval_reader.top_k_em
    reader_top_1_f1 = eval_reader.top_1_f1
    reader_top_k_f1 = eval_reader.top_k_f1
    pipeline_top_1_em = eval_reader.top_1_em_count / total_queries
    pipeline_top_k_em = eval_reader.top_k_em_count / total_queries
    pipeline_top_1_f1 = eval_reader.top_1_f1_sum / total_queries
    pipeline_top_k_f1 = eval_reader.top_k_f1_sum / total_queries

    print("Retriever")
    print("-----------------")
    print(f"total queries: {total_queries}")
    print(f"recall: {retriever_recall}")
    print()
    print("Reader")
    print("-----------------")
    print(f"answer in retrieved docs: {correct_retrieval}")
    print(f"top 1 EM: {reader_top_1_em}")
    print(f"top k EM: {reader_top_k_em}")
    print(f"top 1 F1: {reader_top_1_f1}")
    print(f"top k F1: {reader_top_k_f1}")
    print()
    print("Pipeline")
    print("-----------------")
    print(f"top 1 EM: {pipeline_top_1_em}")
    print(f"top k EM: {pipeline_top_k_em}")
    print(f"top 1 F1: {pipeline_top_1_f1}")
    print(f"top k F1: {pipeline_top_k_f1}")

class EvalRetriever:
    def __init__(self):
        self.outgoing_edges = 1
        self.correct_retrieval = 0
        self.query_count = 0
        self.recall = 0.0
        self.log = []

    def run(self, documents, labels, **kwargs):
        # Open domain mode
        self.query_count += 1
        texts = [x.text for x in documents]
        correct_retrieval = False
        for t in texts:
            for label in labels:
                if label.lower() in t.lower():
                    self.correct_retrieval += 1
                    correct_retrieval = True
                    break
            if correct_retrieval:
                break
        self.recall = self.correct_retrieval / self.query_count
        self.log.append({"documents": documents, "labels": labels, "correct_retrieval": correct_retrieval, **kwargs})
        return {"documents": documents, "labels": labels, "correct_retrieval": correct_retrieval, **kwargs}, "output_1"


class EvalReader:
    def __init__(self):
        self.outgoing_edges = 1
        self.query_count = 0
        self.top_1_em_count = 0
        self.top_k_em_count = 0
        self.top_1_f1_sum = 0
        self.top_k_f1_sum = 0
        self.top_1_em = 0.0
        self.top_k_em = 0.0
        self.top_1_f1 = 0.0
        self.top_k_fi = 0.0

    def run(self, **kwargs):
        if not kwargs:
            return {}, "output_1"
        self.query_count += 1
        predictions = [p["answer"] for p in kwargs["answers"]]
        gold_labels = kwargs["labels"]
        self.top_1_em_count += calculate_em_str_multi(gold_labels, predictions[0])
        self.top_1_f1_sum += calculate_f1_str_multi(gold_labels, predictions[0])
        self.top_k_em_count += max([calculate_em_str_multi(gold_labels, p) for p in predictions])
        self.top_k_f1_sum += max([calculate_f1_str_multi(gold_labels, p) for p in predictions])
        self.update_metrics()
        return {**kwargs}, "output_1"

    def update_metrics(self):
        self.top_1_em = self.top_1_em_count / self.query_count
        self.top_k_em = self.top_k_em_count / self.query_count
        self.top_1_f1 = self.top_1_f1_sum / self.query_count
        self.top_k_f1 = self.top_k_f1_sum / self.query_count


def calculate_em_str_multi(gold_labels, prediction):
    for gold_label in gold_labels:
        result = calculate_em_str(gold_label, prediction)
        if result == 1.0:
            return 1.0
    return 0.0

def calculate_f1_str_multi(gold_labels, prediction):
    results = []
    for gold_label in gold_labels:
        result = calculate_f1_str(gold_label, prediction)
        results.append(result)
    return max(results)