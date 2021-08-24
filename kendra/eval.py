from pathlib import Path
import json
from haystack.eval import calculate_f1_str_multi, calculate_em_str_multi
import pandas as pd
from tqdm import tqdm
from pprint import pprint

INVALID_Q_STR = "INVALID QUESTION FOR KENDRA"

label_dir = Path("../data/nq")
pred_dir = Path(".")
preds_haystack_filename = "predictions_haystack_dpr_7k.json"
preds_kendra_filename = "predictions_kendra_300k.json"
labels_filename = "nq_dev_labels.json"
top_k_eval = 1

def reduce_haystack_preds(preds_haystack_full, top_k_eval):
    ret = {}
    for p in preds_haystack_full:
        query = p["query"]
        answers_str = [a["answer"] for a in p["answers"]]
        ret[query] = answers_str[:top_k_eval]
    return ret

def reduce_kendra_preds(preds_kendra_full):
    ret = {}
    for p in preds_kendra_full:
        query = p["Question"]
        pred_kendra_list = []
        for r in p["ResultItems"]:
            if len(r["AdditionalAttributes"]) > 0:
                for highlights in r['AdditionalAttributes'][0]['Value']['TextWithHighlightsValue']['Highlights']:
                    if highlights['TopAnswer'] == True:
                        answer_text = r['AdditionalAttributes'][0]['Value']['TextWithHighlightsValue']['Text']
                        pred_kendra = answer_text[highlights['BeginOffset']:highlights['EndOffset']]
                        pred_kendra_list.append(pred_kendra)
        ret[query] = pred_kendra_list
    return ret

def reduce_labels(labels):
    ret = {}
    for l in labels:
        query = l["question"]
        ret[query] = l["short_answers"]
    return ret

def join_preds_labels(preds_haystack, preds_kendra, labels):
    ret = {}
    for query in labels:
        pk = preds_kendra.get(query, INVALID_Q_STR)
        ph = preds_haystack[query]
        ret[query] = {"preds_haystack": ph,
                      "preds_kendra": pk,
                      "labels": labels[query]}
    return ret

def generate_results(preds_labels):
    results = []
    for query in tqdm(preds_labels):
        ph = preds_labels[query]["preds_haystack"]
        pk = preds_labels[query]["preds_kendra"]
        l = preds_labels[query]["labels"]
        if len(pk) == 0 or INVALID_Q_STR in pk:
            kendra_pred_available = False
        else:
            kendra_pred_available = True
        if len(l) == 0:
            f1_haystack_top_k = None
            em_haystack_top_k = None
            f1_kendra_top_k = None
            em_kendra_top_k = None
            label_available = False
        else:
            f1_haystack_top_k = max([calculate_f1_str_multi(l, p) for p in ph])
            em_haystack_top_k = max([calculate_em_str_multi(l, p) for p in ph])
            if not kendra_pred_available:
                f1_kendra_top_k = 0
                em_kendra_top_k = 0
            else:
                f1_kendra_top_k = max([calculate_f1_str_multi(l, p) for p in pk])
                em_kendra_top_k = max([calculate_em_str_multi(l, p) for p in pk])
            label_available = True

        d = {"query": query,
             "labels": l,
             "preds_haystack": ph,
             "f1_haystack_top_k": f1_haystack_top_k,
             "em_haystack_top_k": em_haystack_top_k,
             "f1_kendra_top_k": f1_kendra_top_k,
             "em_kendra_top_k": em_kendra_top_k,
             "label_available": label_available,
             "kendra_pred_available": kendra_pred_available,
             "top_k_eval": top_k_eval}
        results.append(d)
    df = pd.DataFrame.from_records(results)
    return df

def main():
    # File loading
    preds_haystack_full = json.load(open(pred_dir/preds_haystack_filename))
    preds_kendra_full = json.load(open(pred_dir/preds_kendra_filename))
    labels_full = json.load(open(label_dir/labels_filename))

    # Data prep
    preds_haystack = reduce_haystack_preds(preds_haystack_full, top_k_eval)
    preds_kendra = reduce_kendra_preds(preds_kendra_full)
    labels = reduce_labels(labels_full)
    preds_labels = join_preds_labels(preds_haystack, preds_kendra, labels)

    # Metric calculation
    df_results = generate_results(preds_labels)

    # Final metric
    df_filtered = df_results[df_results["label_available"] == True]
    valid_questions = len(df_filtered)
    f1_haystack_top_k_1 = df_filtered["f1_haystack_top_k"].mean()
    em_haystack_top_k_1 = df_filtered["em_haystack_top_k"].mean()
    f1_kendra_top_k_1 = df_filtered["f1_kendra_top_k"].mean()
    em_kendra_top_k_1 = df_filtered["em_kendra_top_k"].mean()

    haystack_results_1 = {"f1_haystack_top_k": f1_haystack_top_k_1,
                        "em_haystack_top_k": em_haystack_top_k_1}
    kendra_results_1 = {"f1_kendra_top_k": f1_kendra_top_k_1,
                      "em_kendra_top_k": em_kendra_top_k_1}

    df_filtered_filtered = df_filtered[df_filtered["kendra_pred_available"] == True]
    kendra_valid_answer_count = len(df_filtered_filtered)
    f1_haystack_top_k_2 = df_filtered_filtered["f1_haystack_top_k"].mean()
    em_haystack_top_k_2 = df_filtered_filtered["em_haystack_top_k"].mean()
    f1_kendra_top_k_2 = df_filtered_filtered["f1_kendra_top_k"].mean()
    em_kendra_top_k_2 = df_filtered_filtered["em_kendra_top_k"].mean()

    haystack_results_2 = {"f1_haystack_top_k": f1_haystack_top_k_2,
                        "em_haystack_top_k": em_haystack_top_k_2}
    kendra_results_2 = {"f1_kendra_top_k": f1_kendra_top_k_2,
                      "em_kendra_top_k": em_kendra_top_k_2}

    print(valid_questions)
    print(haystack_results_1)
    print(kendra_results_1)
    print(kendra_valid_answer_count)
    print(haystack_results_2)
    print(kendra_results_2)


if __name__ == "__main__":
    main()